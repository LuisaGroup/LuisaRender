//
// Created by ChenXin on 2022/2/23.
//

#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <base/pipeline.h>
#include <base/grad_integrator.h>

namespace luisa::render {

class MegakernelGradRadiative final : public GradIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelGradRadiative(Scene *scene, const SceneNodeDesc *desc) noexcept
        : GradIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelGradRadiativeInstance final : public GradIntegrator::Instance {

private:
    Pipeline &_pipeline;

private:
    static void _integrate_one_camera(
        Stream &stream, Pipeline &pipeline,
        const Camera::Instance *camera,
        const Filter::Instance *filter,
        uint max_depth,
        uint rr_depth, float rr_threshold,
        compute::Float dLoss_dLi_func(
            Expr<uint2> pixel,
            const Film::Instance *film_rendered,
            const Film::Instance *film_target),
        const Film::Instance *film_rendered,
        const Film::Instance *film_target) noexcept;

public:
    explicit MegakernelGradRadiativeInstance(const MegakernelGradRadiative *node, Pipeline &pipeline) noexcept
        : GradIntegrator::Instance{pipeline, node}, _pipeline{pipeline} {}
    void backpropagation(Stream &stream, luisa::vector<Film::Instance *> film_target,
                         compute::Float dLoss_dLi_func(
                             Expr<uint2> pixel,
                             const Film::Instance *film_rendered,
                             const Film::Instance *film_target)) noexcept override {
        // TODO : create grad buffer

        auto pt = static_cast<const MegakernelGradRadiative *>(node());
        for (auto i = 0u; i < _pipeline.camera_count(); i++) {
            auto [camera, film_rendered, filter] = _pipeline.camera(i);
            // TODO : compare ref with film to get loss
            _integrate_one_camera(
                stream, _pipeline, camera, filter,
                pt->max_depth(), pt->rr_depth(), pt->rr_threshold(),
                dLoss_dLi_func, film_rendered, film_target[i]);
        }
    }
};

unique_ptr<GradIntegrator::Instance> MegakernelGradRadiative::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelGradRadiativeInstance>(this, pipeline);
}

void MegakernelGradRadiativeInstance::_integrate_one_camera(
    Stream &stream, Pipeline &pipeline, const Camera::Instance *camera,
    const Filter::Instance *filter, uint max_depth,
    uint rr_depth, float rr_threshold,
    compute::Float dLoss_dLi_func(
        Expr<uint2> pixel,
        const Film::Instance *film_rendered,
        const Film::Instance *film_target),
    const Film::Instance *film_rendered,
    const Film::Instance *film_target) noexcept {

    auto spp = camera->node()->spp();
    auto image_file = camera->node()->file();
    auto resolution = film_rendered->node()->resolution();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto light_sampler = pipeline.light_sampler();
    auto sampler = pipeline.sampler();
    auto env = pipeline.environment();

    auto command_buffer = stream.command_buffer();
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    using namespace luisa::compute;

    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal,
                                 Float3x3 env_to_world, Float time, Float shutter_weight) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id) + 0.5f;
        auto beta = def(make_float4(1.f));
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        beta *= filter_weight;
        auto swl = SampledWavelengths::sample_visible(sampler->generate_1d());
        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        if (!camera->node()->transform()->is_identity()) {
            camera_ray->set_origin(make_float3(camera_to_world * make_float4(camera_ray->origin(), 1.0f)));
            camera_ray->set_direction(normalize(camera_to_world_normal * camera_ray->direction()));
        }
        beta *= camera_weight;

        auto ray = camera_ray;
        auto pdf_bsdf = def(0.0f);

        // TODO : radiative
        // TODO : the shape of Li (rgb/spectrum)
        auto Li = def(make_float4(1.0f));
        auto d_Li = def(make_float4(0.0f));

        $for(depth, max_depth) {

            // trace
            auto it = pipeline.intersect(ray);

            // miss
            $if(!it->valid()) {
                $break;
            };

            // evaluate material
            auto eta_scale = def(make_float4(1.f));
            auto cos_theta_o = it->wo_local().z;
            auto surface_tag = it->shape()->surface_tag();
            pipeline.dynamic_dispatch_surface(surface_tag, [&](auto surface) {
                // apply alpha map
                auto alpha_skip = def(false);
                if (auto alpha_map = surface->alpha()) {
                    auto alpha = alpha_map->evaluate(*it, swl, time).x;
                    auto u_alpha = sampler->generate_1d();
                    alpha_skip = alpha < u_alpha;
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
                    // create closure
                    auto closure = surface->closure(*it, swl, time);

                    // sample material
                    auto [wi, eval] = closure->sample(*sampler);
                    auto cos_theta_i = dot(wi, it->shading().n());
                    ray = it->spawn_ray(wi);
                    pdf_bsdf = eval.pdf;

                    // radiative bp
                    // TODO : how to change beta from 4 channels to 3 channels
                    closure->backward(wi, shutter_weight * beta * Li);

                    beta *= ite(
                        eval.pdf > 0.0f,
                        eval.f * abs(cos_theta_i) / eval.pdf,
                        make_float4(0.0f));
                    eta_scale = ite(
                        cos_theta_i * cos_theta_o < 0.f &
                            min(eval.alpha.x, eval.alpha.y) < .05f,
                        ite(cos_theta_o > 0.f, sqr(eval.eta), sqrt(1.f / eval.eta)),
                        1.f);
                };
            });

            // rr
            $if(all(beta <= 0.0f)) { $break; };
            auto q = max(swl.cie_y(beta * eta_scale), .05f);
            $if(depth >= rr_depth & q < rr_threshold) {
                $if(sampler->generate_1d() >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
    };
    auto render = pipeline.device().compile(render_kernel);
    auto shutter_samples = camera->node()->shutter_samples();
    stream << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 8u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline.update_geometry(command_buffer, s.point.time)) { dispatch_count = 0u; }
        auto camera_to_world = camera->node()->transform()->matrix(s.point.time);
        auto camera_to_world_normal = transpose(inverse(make_float3x3(camera_to_world)));
        auto env_to_world = env == nullptr || env->node()->transform()->is_identity() ?
                                make_float3x3(1.0f) :
                                transpose(inverse(make_float3x3(
                                    env->node()->transform()->matrix(s.point.time))));
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, camera_to_world, camera_to_world_normal,
                                     env_to_world, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }

    // TODO : update grads

    command_buffer << commit();
    stream << synchronize();
    LUISA_INFO("Backward finished in {} ms.", clock.toc());

    // TODO
    LUISA_ERROR_WITH_LOCATION("unimplemented");
}

}// namespace luisa::render