//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/pipeline.h>
#include <scene/integrator.h>

namespace luisa::render {

class MegakernelPathTracing final : public Integrator {

private:
    uint _max_depth;

public:
    MegakernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "megapath"; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPathTracingInstance final : public Integrator::Instance {

private:
    Pipeline &_pipeline;
    uint _max_depth;

private:
    static void _render_one_camera(
        Stream &stream, Pipeline &pipeline,
        const Camera::Instance *camera,
        const Filter::Instance *filter,
        Film::Instance *film, uint max_depth) noexcept;

public:
    explicit MegakernelPathTracingInstance(const MegakernelPathTracing *node, Pipeline &pipeline) noexcept
        : Integrator::Instance{node}, _pipeline{pipeline}, _max_depth{node->max_depth()} {}
    void render(Stream &stream) noexcept override {
        for (auto i = 0u; i < _pipeline.camera_count(); i++) {
            auto [camera, film, filter] = _pipeline.camera(i);
            _render_one_camera(stream, _pipeline, camera, filter, film, _max_depth);
            film->save(stream, camera->node()->file());
        }
    }
};

unique_ptr<Integrator::Instance> MegakernelPathTracing::build(Pipeline &pipeline, CommandBuffer &) const noexcept {
    return luisa::make_unique<MegakernelPathTracingInstance>(this, pipeline);
}

void MegakernelPathTracingInstance::_render_one_camera(
    Stream &stream, Pipeline &pipeline, const Camera::Instance *camera,
    const Filter::Instance *filter, Film::Instance *film, uint max_depth) noexcept {
    auto spp = camera->node()->spp();
    auto resolution = film->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto light_sampler = pipeline.light_sampler();
    if (light_sampler == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Path tracing cannot render "
            "scenes without lights.");
    }

    auto sampler = pipeline.sampler();
    auto command_buffer = stream.command_buffer();
    film->clear(command_buffer);
    sampler->reset(command_buffer, resolution, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal, Float time) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id);
        auto throughput = def(make_float3(1.0f));
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        throughput *= filter_weight;

        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        camera_ray->set_origin(make_float3(camera_to_world * make_float4(camera_ray->origin(), 1.0f)));
        camera_ray->set_direction(normalize(camera_to_world_normal * camera_ray->direction()));
        throughput *= camera_weight;

        auto ray = camera_ray;
        auto radiance = def(make_float3(0.0f));
        auto pdf_bsdf = def(0.0f);
        $for(depth, max_depth) {
            auto interaction = pipeline.intersect(ray);
            $if(!interaction->valid()) { $break; };
            // evaluate Le
            $if(!interaction->shape()->test_light_flag(Light::property_flag_black)) {
                pipeline.decode_light(interaction->shape()->light_tag(), [&](const Light *light) noexcept {
                    auto eval = light->evaluate(pipeline, *interaction, ray->origin());
                    $if(eval.pdf > 0.0f) {
                        auto pdf_light = eval.pdf * light_sampler->pdf(*interaction);
                        auto mis_weight = ite(depth == 0u, 1.0f, balanced_heuristic(pdf_bsdf, pdf_light));
                        radiance += throughput * eval.Le * mis_weight;
                    };
                });
            };

            $if(interaction->shape()->test_material_flag(Material::property_flag_black)) { $break; };

            // sample light
            Light::Sample light_sample;
            auto light_selection = light_sampler->sample(*sampler, *interaction);
            pipeline.decode_light(light_selection.light_tag, [&](const Light *light) noexcept {
                light_sample = light->sample(pipeline, *sampler, light_selection.instance_id, *interaction);
            });
            auto occluded = pipeline.intersect_any(light_sample.shadow_ray);
            pipeline.decode_material(*interaction, [&](const Material::Closure &material) {
                // evaluate direct lighting
                $if(light_sample.eval.pdf > 0.0f & !occluded) {
                    auto light_pdf = light_sample.eval.pdf * light_selection.pdf;
                    auto wi = light_sample.shadow_ray->direction();
                    auto [f, pdf] = material.evaluate(wi);
                    auto mis_weight = balanced_heuristic(light_pdf, pdf);
                    radiance += throughput * mis_weight * ite(pdf > 0.0f, f, 0.0f) *
                                abs(dot(interaction->shading().n(), wi)) *
                                light_sample.eval.Le / light_pdf;
                };
                // sample material
                auto [wi, eval] = material.sample(*sampler);
                ray = interaction->spawn_ray(wi);
                pdf_bsdf = eval.pdf;
                throughput *= ite(
                    eval.pdf > 0.0f,
                    eval.f * abs(dot(interaction->shading().n(), wi)) / eval.pdf,
                    make_float3());
            });
            // rr
            $if(all(throughput <= 0.0f)) { $break; };
            $if(depth >= 1u) {
                auto l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), throughput);
                auto q = max(l, 0.05f);
                auto r = sampler->generate_1d();
                $if(r >= q) { $break; };
                throughput *= 1.0f / q;
            };
        };
        film->accumulate(pixel_id, radiance);
        sampler->save_state();
    };

    auto render = pipeline.device().compile(render_kernel);
    stream << synchronize();
    Clock clock;
    auto time_start = camera->node()->time_span().x;
    auto time_end = camera->node()->time_span().x;
    auto spp_per_commit = 64u;
    for (auto i = 0u; i < spp; i++) {
        auto t = static_cast<float>((static_cast<double>(i) + 0.5f) / static_cast<double>(spp));
        auto time = lerp(time_start, time_end, t);
        pipeline.update_geometry(command_buffer, time);
        auto camera_to_world = camera->node()->transform()->matrix(t);
        auto camera_to_world_normal = transpose(inverse(make_float3x3(camera_to_world)));
        command_buffer << render(i, camera_to_world, camera_to_world_normal, time).dispatch(resolution);
        if (spp % spp_per_commit == spp_per_commit - 1u) [[unlikely]] { command_buffer << commit(); }
    }
    command_buffer << commit();
    stream << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)
