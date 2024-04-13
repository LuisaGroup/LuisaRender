//
// Created by Jiankai on 2024/3/25.
//
// #include <iostream>
#include "core/logging.h"
#include <luisa-compute.h>
#include <util/imageio.h>
#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <core/stl.h>

namespace luisa::render {

using namespace compute;

class MegakernelPathSpaceDiff final : public DifferentiableIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _max_EPSM_length;

public:
    MegakernelPathSpaceDiff(Scene *scene, const SceneNodeDesc *desc) noexcept
        : DifferentiableIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _max_EPSM_length{std::max(desc->property_uint_or_default("max_EPSM_length", 6u), 6u)} {
          }
    class Matrix{
    private:
        BufferView<float> pool;
        uint size, height, width;
        uint single_mat_size;
    public:
        Matrix(uint batch, uint height, uint width, Pipeline &pipeline):height(height), size(batch), width(width){
            pool = pipeline.create<Buffer<float>>(std::max(batch * height * width, 1u))->view();
            single_mat_size = height*width;
        }
        void set(UInt id, UInt x, UInt y, Float value) {
            pool->write(id*single_mat_size+x*width+y, value);
        }
        Var<float> get(UInt id, UInt x, UInt y) {
            return pool->read(id*single_mat_size+x*width+y);
        }
};
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

};

class MegakernelPathSpaceDiffInstance : public DifferentiableIntegrator::Instance {
public:
    using DifferentiableIntegrator::Instance::Instance;
    [[nodiscard]] virtual Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                                        Expr<uint2> pixel_id, Expr<float> time) const noexcept override;
    void _render_one_camera_backward(
        CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera, Buffer<float> &grad_in) noexcept override;
    
    luisa::unique_ptr<Matrix> mat, mat_param;
};


luisa::unique_ptr<Integrator::Instance> MegakernelPathSpaceDiff::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelPathSpaceDiffInstance>(pipeline, command_buffer, this);
}

Float3 MegakernelPathSpaceDiffInstance::Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept {

    sampler()->start(pixel_id, frame_index);
    auto u_filter = sampler()->generate_pixel_2d();
    auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
    auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
    auto spectrum = pipeline().spectrum();
    auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
    SampledSpectrum beta{swl.dimension(), camera_weight};
    SampledSpectrum Li{swl.dimension()};

    auto ray = camera_ray;
    auto pdf_bsdf = def(1e16f);
    $for(depth, node<MegakernelPathSpaceDiff>()->max_depth()) {

        // trace
        auto wo = -ray->direction();
        auto it = pipeline().geometry()->intersect(ray);

        // miss
        $if(!it->valid()) {
            if (pipeline().environment()) {
                auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
            }
            $break;
        };

        // hit light
        if (!pipeline().lights().empty()) {
            $outline {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                };
            };
        }

        $if(!it->shape().has_surface()) { $break; };

        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_lobe = sampler()->generate_1d();
        auto u_bsdf = sampler()->generate_2d();

        auto u_rr = def(0.f);
        auto rr_depth = node<MegakernelPathSpaceDiff>()->rr_depth();
        $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

        // generate uniform samples
        auto light_sample = LightSampler::Sample::zero(swl.dimension());
        $outline {
            // sample one light
            light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);
        };

        // trace shadow ray
        auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

        // evaluate material
        auto surface_tag = it->shape().surface_tag();
        auto eta_scale = def(1.f);

        $outline {
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](const Surface::Closure *closure) noexcept {
                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) / light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                };
            });
        };

        beta = zero_if_any_nan(beta);
        $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
        auto rr_threshold = node<MegakernelPathSpaceDiff>()->rr_threshold();
        auto q = max(beta.max() * eta_scale, .05f);
        $if(depth + 1u >= rr_depth) {
            $if(q < rr_threshold & u_rr >= q) { $break; };
            beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
        };
    };
    return spectrum->srgb(swl, Li);
}

void MegakernelPathSpaceDiffInstance::_render_one_camera_backward(
    CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera, Buffer<float> & grad_in) noexcept {
    //Path loggers
    auto spp = camera->node()->spp();
    auto resolution = camera->node()->film()->resolution();
    camera->film_grad()->prepare(command_buffer);
    LUISA_INFO("Start backward propagation.");

    

    auto pt = this;
    auto sampler = pt->sampler();
    auto env = pipeline().environment();

    auto pixel_count = resolution.x * resolution.y;
    auto light_sampler = pt->light_sampler();
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << commit() << synchronize();
    auto pt_exact = pt->node<MegakernelPathSpaceDiff>();
    
    //auto pathLogger = PathLogger(pixel_count, node<MegakernelPathSpaceDiff>()->max_depth(), pipeline());
    auto max_depth = pt->node<MegakernelPathSpaceDiff>()->max_depth();

    mat = luisa::make_unique<Matrix>(pixel_count, max_depth * 2, max_depth * 4, pipeline());//for adjoint matrix
    mat_param = luisa::make_unique<Matrix>(pixel_count, max_depth * 2, max_depth * 18, pipeline());
    auto gradient_compute_kernel = _bp_shaders.find(camera);

    if (gradient_compute_kernel == _bp_shaders.end()) {
        using namespace luisa::compute;
        Callable create_local_frame = [](Float3 x) {
            auto normal = normalize(x);
            auto tangent = normalize(cross(normal, Float3(0,1,0)));
            auto bitangent = normalize(cross(normal, tangent));
            return Float3x3(tangent, bitangent, normal);
        };
        Callable inverse_matrix = [&](UInt pixel_id, UInt path_size, UInt max_size) {
            //inverse a matrix which mat->read(i,j) gives the (i,j) element
            auto n = path_size;
            $for (i,2*n) {
                mat->set(pixel_id, i, i+max_size*2, 1);
            };
            $for (i, 2*n) {
                $if (mat->get(pixel_id, i, i) == 0) {
                    auto p = 2*n;
                    $for (j,i + 1,n*2) {
                        $if (mat->get(pixel_id, j, i) != 0) {
                            p=j;
                            $break;
                        };
                    };
                    // 交换行
                    $for(k,n*2)
                    {
                        auto t = mat->get(pixel_id, i, k);
                        mat->set(pixel_id, i, k, mat->get(pixel_id, p,k));
                        mat->set(pixel_id, p, k, t);
                        t = mat->get(pixel_id, i, k+max_size*2);
                        mat->set(pixel_id, i, k+max_size*2, mat->get(pixel_id, p, k+max_size*2));
                        mat->set(pixel_id, p, k+max_size*2, t);
                    };
                };
                $for(j, n*2) {
                    $if(j!=i)
                    {
                        auto factor = mat->get(pixel_id, j, i) / mat->get(pixel_id, i, i);
                        $if(factor==0) {$break;};
                        $for (k,i,n*2) {
                            mat->set(pixel_id, j, k, mat->get(pixel_id, j, k) - factor * mat->get(pixel_id, i, k));
                            mat->set(pixel_id, j, k, mat->get(pixel_id, j, k+max_size*2) - factor * mat->get(pixel_id, i, k+max_size*2));
                        };
                    };
                };
            };
            $for(i, n*2) {
                auto f = mat->get(pixel_id, i, i);
                $for(j, n*2) {
                    auto t = mat->get(pixel_id, i, j+max_size*2);
                    mat->set(pixel_id, i, j, t/f);
                };
            };
        };
        Callable compute_and_scatter_grad = [&](UInt pixel_id, UInt path_size, UInt param_size, ArrayUInt<12> &inst_ids, ArrayUInt<12> &triangle_ids){
            ArrayFloat<24> tmp;
            $for(i, path_size*2){
                tmp[i] = 0.0f;
                $for(j, path_size*2){
                    tmp[i]-=grad_in->read(0)*mat->get(pixel_id, j, i);
                };
            };
            $for(i, param_size) {
                Float grad_tmp = 0.0f;
                $for(j, path_size * 2) {
                    grad_tmp += tmp[j] * mat_param->get(pixel_id, j, i);
                };
                pipeline().differentiation()->add_geom_gradients(grad_tmp, inst_ids[(UInt)(i/6)], triangle_ids[(UInt)(i/6)], i%6);
            };
        };
        Kernel2D _gradient_compute_kernel = [&](UInt frame_index, Float time, Float shutter_weight, ImageFloat Li_1spp, BufferFloat grad_in_real) noexcept {
            //Todo:check block size
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto pixel_id_1d = pixel_id.y * resolution.x + pixel_id.x;
            sampler->start(pixel_id, frame_index);
            auto u_filter = sampler->generate_pixel_2d();
            auto u_lens = camera->node()->requires_lens_sampling() ? sampler->generate_2d() : make_float2(.5f);
            auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
            auto spectrum = pipeline().spectrum();
            auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension()};
            
            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);

            //Logging Path
            ArrayUInt<12> triangle_ids, inst_ids;
            ArrayFloat3<12> bary_coords;
            ArrayFloat<12> etas;
            auto path_size = def(0u);
            bool has_end = false;

            $for(depth, pt->node<MegakernelPathSpaceDiff>()->max_depth()) {
                // trace
                auto wo = -ray->direction();
                auto it = pipeline().geometry()->intersect(ray);
                // miss
                $if(path_size>=12) {$break;};
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        //pathLogger.add_envlight_end(pixel_id_1d, ray->direction(), eval.L, balance_heuristic(pdf_bsdf, eval.pdf));
                    }
                    $break;
                };
                $if(!it->shape().has_surface()) { $break; };
                // hit light
                $if(it->shape().has_light()) {
                    auto eval = light_sampler->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    triangle_ids[path_size] = it->triangle_id();
                    inst_ids[path_size] = it->instance_id();
                    bary_coords[path_size] = it->bary_coord();
                    path_size+=1;
                    has_end=true;
                    $break;
                };


                // sample one light
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();

                auto u_rr = def(0.f);
                auto rr_depth = node<MegakernelPathSpaceDiff>()->rr_depth();
                $if(depth + 1u >= rr_depth) { u_rr = sampler->generate_1d(); };

                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                $outline {
                    // sample one light
                    light_sample = light_sampler->sample(
                        *it, u_light_selection, u_light_surface, swl, time);
                    
                };

                // trace shadow ray
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

                // evaluate material
                auto surface_tag = it->shape().surface_tag();
                auto eta_scale = def(1.f);

                //pathLogger.add_surface_light_end(light_sample, occluded);
                //Todo: log light point, add light sample detail(with position, light tag and ...)
                
                triangle_ids[path_size] = it->triangle_id();
                inst_ids[path_size] = it->instance_id();
                bary_coords[path_size] = it->bary_coord();
                path_size+=1;

                $outline {
                    PolymorphicCall<Surface::Closure> call;
                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                        surface->closure(call, *it, swl, wo, 1.f, time);
                    });
                    call.execute([&](const Surface::Closure *closure) noexcept {
                        // apply opacity map
                        auto alpha_skip = def(false);
                        if (auto o = closure->opacity()) {
                            auto opacity = saturate(*o);
                            alpha_skip = u_lobe >= opacity;
                            u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                        }

                        $if(alpha_skip) {
                            ray = it->spawn_ray(ray->direction());
                            pdf_bsdf = 1e16f;
                        }
                        $else {
                            if (auto dispersive = closure->is_dispersive()) {
                                $if(*dispersive) { swl.terminate_secondary(); };
                            }

                            // direct lighting
                            $if(light_sample.eval.pdf > 0.0f) {
                                auto wi = light_sample.shadow_ray->direction();
                                auto eval = closure->evaluate(wo, wi);
                                auto mis_weight = balance_heuristic(light_sample.eval.pdf, eval.pdf) / light_sample.eval.pdf;
                                $if(!occluded) {
                                    Li += mis_weight * beta * eval.f * light_sample.eval.L;
                                };
                                //pathLogger.add_light(pixel_id_1d, light_sample, closure);
                            };

                            // sample material
                            auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                            ray = it->spawn_ray(surface_sample.wi);
                            pdf_bsdf = surface_sample.eval.pdf;
                            auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                            beta *= w * surface_sample.eval.f;

                            // apply eta scale
                            auto eta = closure->eta().value_or(1.f);
                            etas[path_size-1] = eta;
                            $switch(surface_sample.event) {
                                $case(Surface::event_enter) { eta_scale = sqr(eta); };
                                $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                            };
                        };
                    });
                };

                // rr
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto rr_threshold = node<MegakernelPathSpaceDiff>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, .05f);
                $if(depth + 1u >= rr_depth) {
                    $if(q < rr_threshold & u_rr >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
            $if(!has_end) {
                return;
            };

            // Build Matrix
            {
                auto instance = pipeline().geometry()->instance(inst_ids[0]);
                auto triangle = pipeline().geometry()->triangle(instance, triangle_ids[0]);
                auto v_buffer = instance.vertex_buffer_id();
                auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
                
                auto point_pre_0 = v0->position();
                auto point_pre_1 = v1->position();
                auto point_pre_2 = v2->position();
                auto bary_pre = bary_coords[0];
                
                instance = pipeline().geometry()->instance(inst_ids[1]);
                triangle = pipeline().geometry()->triangle(instance, triangle_ids[1]);
                v_buffer = instance.vertex_buffer_id();
                v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
                
                auto point_cur_0 = v0->position();
                auto point_cur_1 = v1->position();
                auto point_cur_2 = v2->position();
                auto bary_cur = bary_coords[1];

                //need bary coord

                $for(id, 1u, path_size-1){
                    
                    auto normal_cur_0 = v0->normal();
                    auto normal_cur_1 = v1->normal();
                    auto normal_cur_2 = v2->normal();
                    
                    instance = pipeline().geometry()->instance(inst_ids[id + 1]);
                    triangle = pipeline().geometry()->triangle(instance, triangle_ids[id + 1]);
                    v_buffer = instance.vertex_buffer_id();
                    v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                    v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                    v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
                    auto point_nxt_0 = v0->position();
                    auto point_nxt_1 = v1->position();
                    auto point_nxt_2 = v2->position();
                    auto bary_nxt = bary_coords[id+1];

                    $autodiff{
                        requires_grad(point_pre_0, point_pre_1, point_pre_2);
                        requires_grad(point_cur_0, point_cur_1, point_cur_2);
                        requires_grad(point_nxt_0, point_nxt_1, point_nxt_2);
                        requires_grad(bary_pre, bary_cur, bary_nxt);

                        Float3 point_pre = point_pre_0*bary_pre[0]+point_pre_1*bary_pre[1]+point_pre_2*(1-bary_pre[0]-bary_pre[1]);
                        Float3 point_cur = point_cur_0*bary_cur[0]+point_cur_1*bary_cur[1]+point_cur_2*(1-bary_cur[0]-bary_cur[1]);
                        Float3 point_nxt = point_nxt_0*bary_nxt[0]+point_nxt_1*bary_nxt[1]+point_nxt_2*(1-bary_nxt[0]-bary_nxt[1]);
                        Float3 normal_cur = normal_cur_0*bary_cur[0]+normal_cur_1*bary_cur[1]+normal_cur_2*(1-bary_cur[0]-bary_cur[1]);

                        auto trans_mat = create_local_frame(normal_cur);
                        auto wi = normalize(point_pre-point_cur);
                        auto wo = normalize(point_nxt-point_cur);
                        auto wi_local = trans_mat*wi;
                        auto wo_local = trans_mat*wo;
                        auto res = normalize(wi_local+wo_local*etas[id]);   

                        //auto res = wi + wo;
                        //Todo Clear Grad

                        for(int j=0;j<2;j++)
                        {
                            backward(res[j]);
                            auto grad_uv_pre = grad(bary_pre);
                            auto grad_uv_cur = grad(bary_cur);
                            auto grad_uv_nxt = grad(bary_nxt);
                            mat->set(pixel_id_1d, id*2-2+j, 2*id-2, grad_uv_pre[0]);
                            mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 1, grad_uv_pre[1]);
                            mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 0, grad_uv_cur[0]);
                            mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 1, grad_uv_cur[1]);
                            mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 2, grad_uv_nxt[0]);
                            mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 3, grad_uv_nxt[1]);
                            auto point_pre_0_grad = grad(point_pre_0);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 0, point_pre_0_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 1, point_pre_0_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 2, point_pre_0_grad[2]);
                            auto point_pre_1_grad = grad(point_pre_1);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 3, point_pre_1_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 4, point_pre_1_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 5, point_pre_1_grad[2]);
                            auto point_pre_2_grad = grad(point_pre_2);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 6, point_pre_2_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 7, point_pre_2_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 8, point_pre_2_grad[2]);

                            
                            auto point_nxt_0_grad = grad(point_nxt_0);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 0, point_nxt_0_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 1, point_nxt_0_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 2, point_nxt_0_grad[2]);
                            auto point_nxt_1_grad = grad(point_nxt_1);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 3, point_nxt_1_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 4, point_nxt_1_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 5, point_nxt_1_grad[2]);
                            auto point_nxt_2_grad = grad(point_nxt_2);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 6, point_nxt_2_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 7, point_nxt_2_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 8, point_nxt_2_grad[2]);

                            auto point_cur_0_grad = grad(point_cur_0);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 0, point_cur_0_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 1, point_cur_0_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 2, point_cur_0_grad[2]);
                            auto point_cur_1_grad = grad(point_cur_1);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 3, point_cur_1_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 4, point_cur_1_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 5, point_cur_1_grad[2]);
                            auto point_cur_2_grad = grad(point_cur_2);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 6, point_cur_2_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 7, point_cur_2_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 8, point_cur_2_grad[2]);
                            
                            auto normal_cur_0_grad = grad(point_cur_0);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 9, normal_cur_0_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 10, normal_cur_0_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 11, normal_cur_0_grad[2]);
                            auto normal_cur_1_grad = grad(point_cur_1);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 12, normal_cur_1_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 13, normal_cur_1_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 14, normal_cur_1_grad[2]);
                            auto normal_cur_2_grad = grad(point_cur_2);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 15, normal_cur_2_grad[0]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 16, normal_cur_2_grad[1]);
                            mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 17, normal_cur_2_grad[2]);
                        }
                    };
                    point_pre_0 = point_cur_0;
                    point_pre_1 = point_cur_1;
                    point_pre_2 = point_cur_2;
                    bary_pre = bary_cur;
                    point_cur_0 = point_nxt_0;
                    point_cur_1 = point_nxt_1;
                    point_cur_2 = point_nxt_2;
                    bary_cur = bary_nxt;
                };
            }
            inverse_matrix(pixel_id_1d, path_size, max_depth);
            compute_and_scatter_grad(pixel_id_1d, path_size, path_size * 6, inst_ids, triangle_ids);
        };
        gradient_compute_kernel = _bp_shaders.emplace(camera, std::move(pipeline().device().compile(_gradient_compute_kernel))).first;
    }
    auto &&gradient_compute_shader = gradient_compute_kernel->second;
    Clock clock;
    auto sample_id = 0u;

    auto seed_start = node<MegakernelPathSpaceDiff>()->iterations() * spp;
    auto &&Li_1spp = replay_Li[camera];
    auto &&grad_in_real = grad_in;
    auto shutter_samples = camera->node()->shutter_samples();
    for (auto s : shutter_samples) {
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << gradient_compute_shader(seed_start + iteration * spp + sample_id,
                                                 s.point.time, s.point.weight, Li_1spp, grad_in_real)
                                  .dispatch(resolution) << synchronize();
            sample_id++;
            LUISA_INFO("One Iteration finished.");
        }
        #ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            command_buffer << pipeline().printer().retrieve() << synchronize();
        #endif
    }
    auto compute_time = clock.toc();
    LUISA_INFO("gradient compute finished in {} ms", compute_time);
    LUISA_INFO("Start to accumulate gradients.");
    pipeline().differentiation()->accum_gradients(command_buffer);
    command_buffer << commit() << synchronize();
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathSpaceDiff)
