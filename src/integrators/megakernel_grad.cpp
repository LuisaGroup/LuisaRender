//
// Created by ChenXin on 2022/2/23.
//

#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <base/pipeline.h>
#include <base/grad_integrator.h>

namespace luisa::render {

class MegakernelPathTracingGrad final : public GradIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;


public:
    MegakernelPathTracingGrad(Scene *scene, const SceneNodeDesc *desc) noexcept
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

class MegakernelPathTracingGradInstance final : public GradIntegrator::Instance {

private:
    Pipeline &_pipeline;

private:
    static void _integrate_one_camera(
        Stream &stream, Pipeline &pipeline,
        const Camera::Instance *camera,
        const Filter::Instance *filter,
        Film::Instance *film, uint max_depth,
        uint rr_depth, float rr_threshold) noexcept;

public:
    explicit MegakernelPathTracingGradInstance(const MegakernelPathTracingGrad *node, Pipeline &pipeline) noexcept
        : GradIntegrator::Instance{pipeline, node}, _pipeline{pipeline} {}
    void integrate(Stream &stream) noexcept override {
        // TODO : create grad buffer
        // TODO : compare ref with film to get loss

        auto pt = static_cast<const MegakernelPathTracingGrad *>(node());
        for (auto i = 0u; i < _pipeline.camera_count(); i++) {
            auto [camera, film, filter] = _pipeline.camera(i);
            _integrate_one_camera(
                stream, _pipeline, camera, filter, film,
                pt->max_depth(), pt->rr_depth(), pt->rr_threshold());
            film->save(stream, camera->node()->file());
        }
    }
};

unique_ptr<GradIntegrator::Instance> MegakernelPathTracingGrad::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelPathTracingGradInstance>(this, pipeline);
}

void MegakernelPathTracingGradInstance::_integrate_one_camera(
    Stream &stream, Pipeline &pipeline, const Camera::Instance *camera,
    const Filter::Instance *filter, Film::Instance *film, uint max_depth,
    uint rr_depth, float rr_threshold) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = film->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto light_sampler = pipeline.light_sampler();
    auto sampler = pipeline.sampler();
    auto env = pipeline.environment();

    auto command_buffer = stream.command_buffer();
    film->clear(command_buffer);
    sampler->reset(command_buffer, resolution, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal, Float3x3 env_to_world, Float time, Float shutter_weight) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id) + 0.5f;
        auto beta = def(make_float4(1.f));
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        beta *= filter_weight;
        auto swl = SampledWavelengths::sample_visible(sampler->generate_1d());
        auto swl_fixed = swl;
        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        if (!camera->node()->transform()->is_identity()) {
            camera_ray->set_origin(make_float3(camera_to_world * make_float4(camera_ray->origin(), 1.0f)));
            camera_ray->set_direction(normalize(camera_to_world_normal * camera_ray->direction()));
        }
        beta *= camera_weight;

        auto ray = camera_ray;
        // TODO : the shape of Li (rgb/spectrum)
        auto Li = def(make_float4(1.0f));
        auto pdf_bsdf = def(0.0f);

        // TODO : radiative
        auto d_Li = def(make_float4(0.0f));

        $for(depth, max_depth) {

            // trace
            auto it = pipeline.intersect(ray);

            // alpha
            auto alpha = it->alpha();
            auto u_alpha = sampler->generate_1d();
            $if(u_alpha >= alpha) {
                ray = it->spawn_ray(-it->wo());
                pdf_bsdf = 1e16f;
                $continue;
            };

            // evaluate material
            auto eta_scale = def(make_float4(1.f));
            auto cos_theta_o = it->wo_local().z;
            pipeline.decode_material(it->shape()->surface_tag(), *it, swl, time, [&](Surface::Closure &material) {

                // sample material
                auto [wi, eval] = material.sample(*sampler);
                auto cos_theta_i = dot(wi, it->shading().n());
                ray = it->spawn_ray(wi);
                pdf_bsdf = eval.pdf;

                // radiative bp
                material.backward(pipeline, swl_fixed, Li * beta, 1.0f, wi);

                beta *= ite(
                    eval.pdf > 0.0f,
                    eval.f * abs(cos_theta_i) / eval.pdf,
                    make_float4(0.0f));
                swl = eval.swl;
                eta_scale = ite(
                    cos_theta_i * cos_theta_o < 0.f &
                        min(eval.alpha.x, eval.alpha.y) < .05f,
                    ite(cos_theta_o > 0.f, sqr(eval.eta), sqrt(1.f / eval.eta)),
                    1.f);
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
    command_buffer << commit();
    stream << synchronize();
    LUISA_INFO("Backward finished in {} ms.", clock.toc());

    // TODO
    LUISA_ERROR_WITH_LOCATION("unimplemented");
}

}