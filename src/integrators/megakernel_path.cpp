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
    auto env = pipeline.environment();

    auto command_buffer = stream.command_buffer();
    film->clear(command_buffer);
    sampler->reset(command_buffer, resolution, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal,
                                 Float3x3 env_to_world, Float3x3 world_to_env, Float time) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id);
        auto beta = def(make_float3(1.0f));
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        beta *= filter_weight;

        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        if (!camera->node()->transform()->is_identity()) {
            camera_ray->set_origin(make_float3(camera_to_world * make_float4(camera_ray->origin(), 1.0f)));
            camera_ray->set_direction(normalize(camera_to_world_normal * camera_ray->direction()));
        }
        beta *= camera_weight;

        auto ray = camera_ray;
        auto Li = def(make_float3(0.0f));
        auto pdf_bsdf = def(0.0f);
        $for(depth, max_depth) {

            auto add_light_contrib = [&](const Light::Evaluation &eval) noexcept {
                auto mis_weight = ite(depth == 0u, 1.0f, balanced_heuristic(pdf_bsdf, eval.pdf));
                Li += ite(eval.pdf > 0.0f, beta * eval.L * mis_weight, make_float3(0.0f));
            };

            auto env_prob = env == nullptr ? 0.0f : env->selection_prob();

            // trace
            auto it = pipeline.intersect(ray);
            $if(!it->valid()) {
                if (env_prob > 0.0f) {
                    auto w = ray->direction();
                    if (env != nullptr && !env->node()->transform()->is_identity()) {
                        w = world_to_env * w;
                    }
                    auto eval = env->evaluate(w, time);
                    eval.pdf *= env_prob;
                    add_light_contrib(eval);
                }
                $break;
            };

            // evaluate Le
            $if(it->shape()->has_light()) {
                auto eval = light_sampler->evaluate(
                    *it, ray->origin(), time);
                eval.pdf *= 1.0f - env_prob;
                add_light_contrib(eval);
            };

            // sample one light
            $if(!it->shape()->has_material()) { $break; };
            Light::Sample light_sample;
            if (env_prob > 0.0f) {
                auto u = sampler->generate_1d();
                $if(u < env_prob) {
                    light_sample = env->sample(*sampler, *it, time);
                    light_sample.eval.pdf *= env_prob;
                    if (env != nullptr && !env->node()->transform()->is_identity()) {
                        light_sample.shadow_ray->set_direction(
                            env_to_world * light_sample.shadow_ray->direction());
                    }
                }
                $else {
                    light_sample = light_sampler->sample(*sampler, *it, time);
                    light_sample.eval.pdf *= 1.0f - env_prob;
                };
            } else {
                light_sample = light_sampler->sample(*sampler, *it, time);
                light_sample.eval.pdf *= 1.0f - env_prob;
            }

            // trace shadow ray
            auto occluded = pipeline.intersect_any(light_sample.shadow_ray);

            // evaluate material
            pipeline.decode_material(it->shape()->material_tag(), *it, [&](const Material::Closure &material) {
                // direct lighting
                $if(light_sample.eval.pdf > 0.0f & !occluded) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto [f, pdf] = material.evaluate(wi, time);
                    auto mis_weight = balanced_heuristic(light_sample.eval.pdf, pdf);
                    Li += beta * mis_weight * ite(pdf > 0.0f, f, 0.0f) *
                          abs(dot(it->shading().n(), wi)) *
                          light_sample.eval.L / light_sample.eval.pdf;
                };

                // sample material
                auto [wi, eval] = material.sample(*sampler, time);
                ray = it->spawn_ray(wi);
                pdf_bsdf = eval.pdf;
                beta *= ite(
                    eval.pdf > 0.0f,
                    eval.f * abs(dot(it->shading().n(), wi)) / eval.pdf,
                    make_float3(0.0f));
            });

            // rr
            $if(all(beta <= 0.0f)) { $break; };
            auto lum = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
            $if(depth >= 1u | lum < 0.95f) {
                $if(sampler->generate_1d() >= lum) { $break; };
                beta *= 1.0f / lum;
            };
        };
        film->accumulate(pixel_id, Li);
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
        auto env_to_world = env == nullptr ?
                                make_float3x3(1.0f) :
                                transpose(inverse(make_float3x3(env->node()->transform()->matrix(time))));
        auto world_to_env = inverse(env_to_world);
        command_buffer << render(i, camera_to_world, camera_to_world_normal, env_to_world, world_to_env, time)
                              .dispatch(resolution);
        if (spp % spp_per_commit == spp_per_commit - 1u) [[unlikely]] {
            command_buffer << commit();
        }
    }
    command_buffer << commit();
    stream << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)
