//
// Created by ChenXin on 2022/2/23.
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
    
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
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
    
    class Matrix{
        private:
            Buffer<float> pool;
            uint size, height, width;
            uint single_mat_size;
        public:
            Matrix(uint batch, uint height, uint width, Pipeline &pipeline):height(height), size(batch), width(width){
                auto &&device = pipeline().device();
                pool = device.create_buffer<float>(batch*size*width);
                single_mat_size = height*width;
            }
            void set(uint id, uint x, uint y, float value){
                pool.write(id*single_mat_size+x*width+y, value);
            }
            void get(uint id, uint x, uint y){
                return pool.read(id*single_mat_size+x*width+y);
            }
            void inverse(){
                //inverse the matrix in place
            }
        };
    class PathLogger{
    private:
        Buffer<float3> vertexes;
        Buffer<float3> normals;
        Buffer<float3> colors;
        Buffer<float2> uvs;
        Buffer<uint> inst_ids;
        Buffer<uint> triangle_ids;
        Buffer<uint> surface_tags;//Currently not use
        Buffer<uint> path_sizes;
    
        Buffer<bool> has_end;
        Buffer<float3> light;
        unique_ptr<Matrix> mat, mat_param;
        Buffer<float> param_grad;
        float3 light_grad;
        uint max_size, max_depth;

        Pipeline &_pipeline;

        Buffer<Triangle> triangle_ids;
        Buffer<uint> inst_ids;

    public:
        PathLogger(uint max_size, uint max_depth, Pipeline &pipeline):max_size(max_size), max_depth(max_depth){
            auto &&device = pipeline.device();
            vertexes = device.create_buffer<float3>(max_size*max_depth*3);
            normals = device.create_buffer<float3>(max_size*max_depth*3);
            uvs = device.create_buffer<float2>(max_size*max_depth);
            inst_ids = device.create_buffer<uint>(max_size*max_depth);
            triangle_ids = device.create_buffer<uint>(max_size*max_depth);
            surface_tags = device.create_buffer<uint>(max_size*max_depth);
            path_sizes = device.create_buffer<uint>(max_size);
            
            has_end = device.create_buffer<bool>(max_size);
            light = device.create_buffer<float3>(max_size*max_depth);

            param_grad = device.create_buffer<float>(max_size*max_depth*18);
            mat = luisa::make_unique<Matrix>(max_size, max_depth*2, max_depth*4, pipeline);//for adjoint matrix
            mat_param = luisa::make_unique<Matrix>(max_size, max_depth*2, max_depth*18, pipeline);

            _pipeline = pipeline;

        }
        void add_vertex(Expr<uint> pixel_id, Expr<float3> vertex, Expr<float3> normal, Expr<float2> uv, Expr<uint> inst_id, Expr<uint> triangle_id, Expr<uint> surface_tag){
            
            auto cur_size = path_sizes->read(pixel_id);
            auto st_point = pixel_id*max_depth+cur_size;
            inst_ids->write(st_point, inst_id);
            triangle_ids->write(st_point, triangle_id);

            // compute dpdu and dpdv
            // auto duv0 = uv1 - uv0;
            // auto duv1 = uv2 - uv0;
            // auto det = duv0.x * duv1.y - duv0.y * duv1.x;
            // auto inv_det = 1.f / det;
            // auto dp0_local = p1_local - p0_local;
            // auto dp1_local = p2_local - p0_local;
            // auto dpdu_local = (dp0_local * duv1.y - dp1_local * duv0.y) * inv_det;
            // auto dpdv_local = (dp1_local * duv0.x - dp0_local * duv1.x) * inv_det;

            // // world space
            // auto m = make_float3x3(shape_to_world);
            // auto t = make_float3(shape_to_world[3]);
            // auto p = m * interpolate(bary, p0_local, p1_local, p2_local) + t;

            // auto c = cross(m * dp0_local, m * dp1_local);
            // auto area = length(c) * .5f;
            // auto ng = normalize(c);
            // auto fallback_frame = Frame::make(ng);
            // auto dpdu = ite(det == 0.f, fallback_frame.s(), m * dpdu_local);
            // auto dpdv = ite(det == 0.f, fallback_frame.t(), m * dpdv_local);
            // auto mn = transpose(inverse(m));
            // auto ns = ite(instance.has_vertex_normal(), normalize(mn * ns_local), ng);
            // auto uv = ite(instance.has_vertex_uv(), interpolate(bary, uv0, uv1, uv2), bary.yz());
            // return {.g = {.p = p,
            //             .n = ng,
            //             .area = area},
            //         .ps = p,
            //         .ns = face_forward(ns, ng),
            //         .dpdu = dpdu,
            //         .dpdv = dpdv,
            //         .uv = uv};
        }
        void add_envlight_end(Expr<float3> beta, Expr<float3> L, Expr<float> pdf){
            has_end = true
            light = beta*L*pdf;
        }
        float3x3 create_local_frame(float3 normal){
            auto tangent = normalize(cross(normal, float3(0,1,0)));
            auto bitangent = normalize(cross(normal, tangent));
            return float3x3(tangent, bitangent, normal);
        }
        void matMulti(Expr<uint> pixel_id, Buffer<float2> grad_uv, Buffer<float> mat, Buffer<float> mat_param){
            auto path_size = sizes->read(pixel_id);
            auto res = make_float2(0,0);
            for(int i=0;i<path_size;i++){
                auto dLdm = 0.0;
                for(int j=0;j<path_size;j++)
                    dLdm+=grad_uv->read(pixel_id) * mat->read_adj(pixel_id, i, j);
                for(int j=0;j<path_param_size;j++)
                    param_grad->write(pixel_id, i,j, dLdm*mat_param->read(pixel_id, i, j)+param_grad->read(pixel_id, i, j));
            }
        }
        uint get_indx(Expr<uint> pixel_id, Expr<uint> i, Expr<uint> j){
            return pixel_id*max_size*3+i*3+j;
        }
        void scatter_grad(Expr<uint> pixel_id, luisa::vector<Buffer<float>> &grad_buffer, luisa::vector<Int> &pos_mapping, luisa::vector<Int> &normal_mapping)
        {
            auto path_size = param_sizes->read(pixel_id);
            for(int i=0;i<path_size;i++){
                auto pos_grad_idx = pos_mapping[inst_ids->read(pixel_id,i)];
                auto normal_grad_idx = normal_mapping[inst_ids->read(pixel_id,i)];
                if (pos_grad_idx!=-1){
                    for(int j=0;j<3;j++)
                        grad_buffer[pos_grad_idx].write(get_indx(pixel_id, i, j), make_float3(param_grad->read(pixel_id*max_size*18+i*18+j*3+0), param_grad->read(pixel_id*max_size*18+i*18+j*3+1),param_grad->read(pixel_id*max_size*18+i*18+j*3+2)));
                }
                if (normal_grad_idx!=-1){
                    for(int j=0;j<3;j++)
                        grad_buffer[normal_grad_idx].write(get_indx(pixel_id, i, j), make_float3(param_grad->read(pixel_id*max_size*18+i*18+j*3+9), param_grad->read(pixel_id*max_size*18+i*18+j*3+10),param_grad->read(pixel_id*max_size*18+i*18+j*3+11)));
                }
            }
        }
        void compute_gradients(Buffer<float> &grad_in)
        {
            Kernel2D build_matrix = [&](UInt pixel_id) noexcept {
                auto grad = grad_in.read(pixel_id);
                $if(!has_end) return;
                auto _size = path_sizes->read(pixel_id);
                for(uint id = 1; id < _size-1; id++){
                    float3 point_nxt_0, point_nxt_1, point_nxt_2;
                    float3 point_pre_0, point_pre_1, point_pre_2;
                    float3 point_cur_0, point_cur_1, point_cur_2;
                    float3 normal_cur, normal_cur_0, normal_cur_1, normal_cur_2;
                    float2 uv_pre, uv_cur, uv_nxt;

                    auto instance = _pipeline.geometry()->instance(inst_id);
                    auto triangle = _pipeline.geometry()->triangle(instance, triangle_id);
                    auto v_buffer = instance.vertex_buffer_id();
                    auto v0 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
                    auto v1 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
                    auto v2 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
                    
                    auto p0_local = v0->position();
                    auto p1_local = v1->position();
                    auto p2_local = v2->position();
                    auto ns_local = interpolate(bary, v0->normal(), v1->normal(), v2->normal());

                    auto uv0 = v0->uv();
                    auto uv1 = v1->uv();
                    auto uv2 = v2->uv();

                    float2 uv_pre = readuv(uvs, id-1, pixel_id);
                    float2 uv_cur = readuv(uvs, id, pixel_id);
                    float2 uv_nxt = readuv(uvs, id+1, pixel_id);

                    
                    float3 point_pre_0 = readpoint(vertexes, id-1, 0, pixel_id);
                    float3 point_pre_1 = readpoint(vertexes, id-1, 1, pixel_id);
                    float3 point_pre_2 = readpoint(vertexes, id-1, 2, pixel_id);
                    
                    float3 point_cur_0 = readpoint(vertexes, id, 0, pixel_id);
                    float3 point_cur_1 = readpoint(vertexes, id, 1, pixel_id);
                    float3 point_cur_2 = readpoint(vertexes, id, 2, pixel_id);
                    
                    float3 point_nxt_0 = readpoint(vertexes, id+1, 0, pixel_id);
                    float3 point_nxt_1 = readpoint(vertexes, id+1, 1, pixel_id);
                    float3 point_nxt_2 = readpoint(vertexes, id+1, 2, pixel_id);

                    
                    float3 normal_cur_0 = readpoint(normals, id, 0, pixel_id);
                    float3 normal_cur_1 = readpoint(normals, id, 1, pixel_id);
                    float3 normal_cur_2 = readpoint(normals, id, 2, pixel_id);


                    uv_cur = reads(uvs, id, pixel_id);
                    uv_nxt = reads(uvs, id+1, pixel_id);

                    $autodiff{
                        
                        float3 point_pre = point_pre_0*uv_pre[0]+point_pre_1*uv_pre[1]+point_pre_2*(1-uv_pre[0]-uv_pre[1]);
                        float3 point_cur = point_cur_0*uv_cur[0]+point_cur_1*uv_cur[1]+point_cur_2*(1-uv_cur[0]-uv_cur[1]);
                        float3 point_nxt = point_nxt_0*uv_nxt[0]+point_nxt_1*uv_nxt[1]+point_nxt_2*(1-uv_nxt[0]-uv_nxt[1]);
                        float3 normal_cur = normal_cur_0*uv_cur[0]+normal_cur_1*uv_cur[1]+normal_cur_2*(1-uv_cur[0]-uv_cur[1]);

                        auto Mat = create_local_frame(normal_cur);
                        auto wi = normalize(point_pre-point_cur);
                        auto wo = normalize(point_nxt-point_cur);
                        auto wi_local = Mat*wi;
                        auto wo_local = Mat*wo;

                        auto res = normalize(wi_local+wo_local*etas[id]);   
                    //Todo Clear Grad
                        for(int j=0;j<2;j++)
                        {
                            backward(res[j]);
                            auto grad_uv_pre = grad(uv_pre);
                            auto grad_uv_cur = grad(uv_cur);
                            auto grad_uv_nxt = grad(uv_nxt);
                            mat->set(pixel_id, id*2-2+j, 2*id-2, grad_uv_pre[0]);
                            mat->set(pixel_id, id*2-2+j, 2*id-1, grad_uv_pre[1]);
                            mat->set(pixel_id, id*2-2+j, 2*id-0, grad_uv_cur[0]);
                            mat->set(pixel_id, id*2-2+j, 2*id+1, grad_uv_cur[1]);
                            mat->set(pixel_id, id*2-2+j, 2*id+2, grad_uv_nxt[0]);
                            mat->set(pixel_id, id*2-2+j, 2*id+3, grad_uv_nxt[1]);
                            auto point_pre_0_grad = grad(point_pre_0);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+0, point_pre_0_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+1, point_pre_0_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+2, point_pre_0_grad[2]);
                            auto point_pre_1_grad = grad(point_pre_1);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+3, point_pre_1_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+4, point_pre_1_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+5, point_pre_1_grad[2]);
                            auto point_pre_2_grad = grad(point_pre_2);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+6, point_pre_2_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+7, point_pre_2_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id-1)+8, point_pre_2_grad[2]);

                            
                            auto point_nxt_0_grad = grad(point_nxt_0);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+0, point_nxt_0_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+1, point_nxt_0_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+2, point_nxt_0_grad[2]);
                            auto point_nxt_1_grad = grad(point_nxt_1);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+3, point_nxt_1_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+4, point_nxt_1_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+5, point_nxt_1_grad[2]);
                            auto point_nxt_2_grad = grad(point_nxt_2);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+6, point_nxt_2_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+7, point_nxt_2_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id+1)+8, point_nxt_2_grad[2]);

                            auto point_cur_0_grad = grad(point_cur_0);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+0, point_cur_0_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+1, point_cur_0_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+2, point_cur_0_grad[2]);
                            auto point_cur_1_grad = grad(point_cur_1);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+3, point_cur_1_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+4, point_cur_1_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+5, point_cur_1_grad[2]);
                            auto point_cur_2_grad = grad(point_cur_2);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+6, point_cur_2_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+7, point_cur_2_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+8, point_cur_2_grad[2]);
                            
                            auto normal_cur_0_grad = grad(point_cur_0);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+9, normal_cur_0_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+10, normal_cur_0_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+11, normal_cur_0_grad[2]);
                            auto normal_cur_1_grad = grad(point_cur_1);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+12, normal_cur_1_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+13, normal_cur_1_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+14, normal_cur_1_grad[2]);
                            auto normal_cur_2_grad = grad(point_cur_2);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+15, normal_cur_2_grad[0]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+16, normal_cur_2_grad[1]);
                            mat_param->set(pixel_id, id*2-2+j, 18*(id)+17, normal_cur_2_grad[2]);
                        }
                    }
                }
            };
            Kernel2D inverse_matrix = [&](UInt pixel_id) noexcept {
                //inverse a matrix which mat->read(i,j) gives the (i,j) element
                auto n = sizes->read(pixel_id);
                $for (auto i = 0; i < n*2; i++) {
                    mat->write_adj(pixel_id, i, i, 1);
                }
                $for (auto i = 0; i < n; i++) {
                    $if (mat->read(pixel_id, i, i) == 0) {
                        auto j=0,k=0;
                        $for (j = i + 1; j < n; ++j) {
                            $if (mat->read(pixel_id, j, i) != 0) {
                                break;
                            }
                        }
                        if (j == n) {
                            invalid->write(pixel_id, true);
                        }
                        // 交换行
                        for(k=0;k<n;k++)
                        {
                            auto t = mat->read(pixel_id, i, k);
                            mat->write(pixel_id, i, k, mat->read(pixel_id, j,k));
                            mat->write(pixel_id, j, k, t);
                            auto t = mat->read_adj(pixel_id, i, k);
                            mat->write_adj(pixel_id, i, k, mat->read_adj(pixel_id, j,k));
                            mat->write_adj(pixel_id, j, k, t);
                        }
                    }
                    for (size_t j = i + 1; j < n; ++j) {
                        double factor = mat->read(pixel_id, j, i) / result->read(pixel_id, i, i);
                        for (size_t k = i; k < n; ++k) {
                            mat->write(pixel_id, j, k, mat->read(pixel_id, j, k) - factor * mat->read(pixel_id, i, k));
                        }
                        for (size_t k = 0; k < n; ++k) {
                            mat->write_adj(pixel_id, j, k, mat->read_adj(pixel_id, j, k) - factor * mat->read_adj(i, k));
                        }
                    }
                }
            };


            Kernel2D compute_gradient_kernel = [&](UInt frame_index, Buffer<float> grad_in, UInt grad_type) noexcept {
                auto pixel_id = dispatch_id().xy();
                auto grad_pos = make_float2(grad_in->read(pixel_id*5+3),grad_in->read(pixel_id*5+4));
                auto grad_color = make_float3(grad_in->read(pixel_id*5+0),grad_in->read(pixel_id*5+1),grad_in->read(pixel_id*5+2));
                matMulti(pixel_id, grad_uv, mat, mat_param);
                scatter_grad(pixel_id);
                //auto color2theta = 
            };
        }
        void reset() {
            sizes->clear();
        }
    };
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
    $for(depth, node<MegakernelReplayDiff>()->max_depth()) {

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
        auto rr_depth = node<MegakernelReplayDiff>()->rr_depth();
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
        auto rr_threshold = node<MegakernelReplayDiff>()->rr_threshold();
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
    auto pt_exact = pt->node<MegakernelReplayDiff>();
    
    
    auto pathLogger = PathLogger(pixel_count, node<MegakernelPathSpaceDiff>()->max_depth(), pipeline());

    auto gradient_compute_kernel = compute_kernels.find(camera);
    if (gradient_compute_kernel == compute_kernels.end()) {
        using namespace luisa::compute;
        Kernel2D _tracing_and_logging_kernel = [&](UInt frame_index, Float time, Float shutter_weight, ImageFloat Li_1spp) noexcept {
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
            $for(depth, pt->node<MegakernelReplayDiff>()->max_depth()) {
                // trace
                auto wo = -ray->direction();
                auto it = pipeline().geometry()->intersect(ray);
                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        pathLogger.add_envlight_end(pixel_id_1d, ray->direction(), eval.L, balance_heuristic(pdf_bsdf, eval.pdf));
                    }
                    $break;
                };
                $if(!it->shape().has_surface()) { $break; };
                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape().has_light()) {
                        auto eval = light_sampler->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        pathLogger.add_surface_light_end(pixel_id_1d, it);
                        $break;
                    };
                }


                // sample one light
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();

                auto u_rr = def(0.f);
                auto rr_depth = node<MegakernelReplayDiff>()->rr_depth();
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
                                $if(!occluded)
                                {
                                    Li += mis_weight * beta * eval.f * light_sample.eval.L;
                                }
                                pathLogger.add_light(pixel_id_1d, light_sample, closure);
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
                            pathLogger.add_vertex(pixel_id_1d, it, closure);
                        };
                    });
                };

                // rr
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto rr_threshold = node<MegakernelReplayDiff>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, .05f);
                $if(depth + 1u >= rr_depth) {
                    $if(q < rr_threshold & u_rr >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
            pathlogger.compute_gradients();
        };
        gradient_compute_kernel = gradient_compute_kernels.emplace(camera, std::move(pipeline().device().compile(_gradient_compute_kernel))).first;
    }
    auto &&gradient_compute_shader = gradient_compute_kernel->second;

    Clock clock;
    auto sample_id = 0u;

    auto seed_start = node<MegakernelReplayDiff>()->iterations() * spp;
    auto &&Li_1spp = replay_Li[camera];
    auto &&grad_in_real = grad_in;
    auto shutter_samples = camera->node()->shutter_samples();

    command_buffer << gradient_compute_shader(seed_start, grad_in_real) .dispatch(resolution) << synchronize();

    auto compute_time = clock.toc();
    LUISA_INFO("gradient compute finished in {} ms", compute_time);
    LUISA_INFO("Start to accumulate gradients.");
    pipeline().differentiation()->accum_gradients(command_buffer);
    command_buffer << commit() << synchronize();
    LUISA_INFO("Step finished in {} ms", clock.toc()-bp_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathSpaceDiff)
