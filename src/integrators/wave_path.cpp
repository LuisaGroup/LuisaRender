//
// Created by Mike Smith on 2022/1/10.
//

#include <tinyexr.h>
#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class WavefrontPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    WavefrontPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] bool differentiable() const noexcept override { return false; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class WavefrontPathTracingInstance final : public Integrator::Instance {

private:
    Pipeline &_pipeline;
    luisa::vector<float4> _pixels;

private:
    static void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline,
        WavefrontPathTracingInstance *pt, Camera::Instance *camera) noexcept;

public:
    explicit WavefrontPathTracingInstance(const WavefrontPathTracing *node, Pipeline &pipeline) noexcept
        : Integrator::Instance{pipeline, node}, _pipeline{pipeline} {}

    void render(Stream &stream) noexcept override {
        auto pt = node<WavefrontPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < _pipeline.camera_count(); i++) {
            auto camera = _pipeline.camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _pixels.resize(next_pow2(pixel_count) * 4u);
            _render_one_camera(command_buffer, _pipeline, this, camera);
            camera->film()->download(command_buffer, _pixels.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();
            if (film_path.extension() != ".exr") [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unexpected film file extension. "
                    "Changing to '.exr'.");
                film_path.replace_extension(".exr");
            }
            auto size = make_int2(resolution);
            const char *err = nullptr;
            SaveEXR(reinterpret_cast<const float *>(_pixels.data()),
                    size.x, size.y, 4, false, film_path.string().c_str(), &err);
            if (err != nullptr) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to save film to '{}'.",
                    film_path.string());
            }
        }
    }
};

unique_ptr<Integrator::Instance> WavefrontPathTracing::build(Pipeline &pipeline, CommandBuffer &) const noexcept {
    return luisa::make_unique<WavefrontPathTracingInstance>(this, pipeline);
}

void WavefrontPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline,
    WavefrontPathTracingInstance *pt, Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto light_sampler = pipeline.light_sampler();
    auto sampler = pipeline.sampler();
    auto env = pipeline.environment();

    camera->film()->clear(command_buffer);
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 env_to_world,
                                 Float time, Float shutter_weight) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto [camera_ray, camera_weight] = camera->generate_ray(
            *sampler, pixel_id, time, camera_to_world);
        auto beta = def(make_float4(camera_weight));
        auto swl = SampledWavelengths::sample_visible(sampler->generate_1d());
        auto ray = camera_ray;
        auto Li = def(make_float4(0.0f));
        auto pdf_bsdf = def(0.0f);
        $for(depth, pt->node<WavefrontPathTracing>()->max_depth()) {

            auto add_light_contrib = [&](const Light::Evaluation &eval) noexcept {
                auto mis_weight = ite(depth == 0u, 1.0f, balanced_heuristic(pdf_bsdf, eval.pdf));
                Li += ite(eval.pdf > 0.0f, beta * eval.L * mis_weight, make_float4(0.0f));
            };

            auto env_prob = env == nullptr ? 0.0f : env->selection_prob();

            // trace
            auto it = pipeline.intersect(ray);

            // miss
            $if(!it->valid()) {
                if (env_prob > 0.0f) {
                    auto eval = env->evaluate(ray->direction(), env_to_world, swl, time);
                    eval.L /= env_prob;
                    add_light_contrib(eval);
                }
                $break;
            };

            // hit light
            if (light_sampler != nullptr && env_prob < 1.f) {
                $if(it->shape()->has_light()) {
                    auto eval = light_sampler->evaluate(*it, ray->origin(), swl, time);
                    eval.L /= 1.0f - env_prob;
                    add_light_contrib(eval);
                };
            }

            // sample one light
            $if(!it->shape()->has_surface()) { $break; };
            Light::Sample light_sample;
            if (env_prob > 0.0f) {
                auto u = sampler->generate_1d();
                $if(u < env_prob) {
                    light_sample = env->sample(*sampler, it->p(), env_to_world, swl, time);
                    light_sample.eval.pdf *= env_prob;
                }
                $else {
                    if (light_sampler != nullptr) {
                        light_sample = light_sampler->sample(*sampler, *it, swl, time);
                        light_sample.eval.pdf *= 1.0f - env_prob;
                    }
                };
            } else if (light_sampler != nullptr) {
                light_sample = light_sampler->sample(*sampler, *it, swl, time);
                light_sample.eval.pdf *= 1.0f - env_prob;
            }

            // trace shadow ray
            auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
            auto occluded = pipeline.intersect_any(shadow_ray);

            // evaluate material
            auto eta_scale = def(make_float4(1.f));
            auto cos_theta_o = it->wo_local().z;
            auto surface_tag = it->shape()->surface_tag();
            pipeline.dynamic_dispatch_surface(surface_tag, [&](auto surface) {
                // apply normal map
                if (auto normal_map = surface->normal()) {
                    auto normal_local = 2.f * normal_map->evaluate(*it, swl, time).value.xyz() - 1.f;
                    auto normal = it->shading().local_to_world(normal_local);
                    it->set_shading(Frame::make(normal, it->shading().u()));
                }
                // apply alpha map
                auto alpha_skip = def(false);
                if (auto alpha_map = surface->alpha()) {
                    auto alpha = alpha_map->evaluate(*it, swl, time).value.x;
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

                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.wi;
                        auto eval = closure->evaluate(wi);
                        auto cos_theta_i = dot(it->shading().n(), wi);
                        auto is_trans = cos_theta_i * cos_theta_o < 0.f;
                        auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                        Li += beta * mis_weight * ite(eval.pdf > 0.0f, eval.f, 0.0f) *
                              abs_dot(it->shading().n(), wi) *
                              light_sample.eval.L / light_sample.eval.pdf;
                    };

                    // sample material
                    auto [wi, eval] = closure->sample(*sampler);
                    auto cos_theta_i = dot(wi, it->shading().n());
                    ray = it->spawn_ray(wi);
                    pdf_bsdf = eval.pdf;
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
            auto rr_depth = pt->node<WavefrontPathTracing>()->rr_depth();
            auto rr_threshold = pt->node<WavefrontPathTracing>()->rr_threshold();
            $if(depth >= rr_depth & q < rr_threshold) {
                $if(sampler->generate_1d() >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        camera->film()->accumulate(pixel_id, swl.srgb(Li * shutter_weight));
    };
    auto render = pipeline.device().compile(render_kernel);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 16u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline.update_geometry(command_buffer, s.point.time)) {
            dispatch_count = 0u;
        }
        auto camera_to_world = camera->node()->transform()->matrix(s.point.time);
        auto env_to_world = env == nullptr || env->node()->transform()->is_identity() ?
                                make_float3x3(1.0f) :
                                transpose(inverse(make_float3x3(
                                    env->node()->transform()->matrix(s.point.time))));
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, camera_to_world, env_to_world,
                                     s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }
    command_buffer << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WavefrontPathTracing)
