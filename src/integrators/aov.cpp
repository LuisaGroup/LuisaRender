//
// Created by Mike Smith on 2022/1/10.
//

#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/scene.h>

namespace luisa::render {

using namespace compute;

static const luisa::unordered_map<luisa::string_view, uint>
    aov_component_to_channels{{"sample", 3u},
                              {"diffuse", 3u},
                              {"specular", 3u},
                              {"normal", 3u},
                              {"albedo", 3u},
                              {"depth", 1u},
                              {"roughness", 2u},
                              {"ndc", 3u},
                              {"mask", 1u}};

class AuxiliaryBufferPathTracing final : public Integrator {

public:
    enum struct DumpStrategy {
        POWER2,
        ALL,
        FINAL,
    };

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _noisy_count;
    DumpStrategy _dump_strategy{};
    luisa::unordered_set<luisa::string> _enabled_aov;

public:
    AuxiliaryBufferPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _noisy_count{std::max(desc->property_uint_or_default("noisy_count", 8u), 8u)} {
        auto components = desc->property_string_list_or_default("components", {"all"});
        for (auto &comp : components) {
            for (auto &c : comp) { c = static_cast<char>(std::tolower(c)); }
            if (comp == "all") {
                for (auto &[name, _] : aov_component_to_channels) {
                    _enabled_aov.emplace(name);
                }
            } else if (aov_component_to_channels.contains(comp)) {
                _enabled_aov.emplace(comp);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Ignoring unknown AOV component '{}'. [{}]",
                    comp, desc->source_location().string());
            }
        }
        for (auto &&comp : _enabled_aov) {
            LUISA_INFO("Enabled AOV component '{}'.", comp);
        }
        auto dump = desc->property_string_or_default("dump", "power2");
        for (auto &c : dump) { c = static_cast<char>(std::tolower(c)); }
        if (dump == "all") {
            _dump_strategy = DumpStrategy::ALL;
        } else if (dump == "final") {
            _dump_strategy = DumpStrategy::FINAL;
        } else {
            if (dump != "power2") [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown dump strategy '{}'. "
                    "Fallback to power2 strategy. [{}]",
                    dump, desc->source_location().string());
            }
            _dump_strategy = DumpStrategy::POWER2;
        }
    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto noisy_count() const noexcept { return _noisy_count; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto dump_strategy() const noexcept { return _dump_strategy; }
    [[nodiscard]] auto is_component_enabled(luisa::string_view component) const noexcept {
        return _enabled_aov.contains(component);
    }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class AuxiliaryBufferPathTracingInstance final : public Integrator::Instance {

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    luisa::optional<Window> _window;

private:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

public:
    explicit AuxiliaryBufferPathTracingInstance(
        const AuxiliaryBufferPathTracing *node,
        Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<AuxiliaryBufferPathTracing>();
        CommandBuffer command_buffer{&stream};
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, camera);
            command_buffer << compute::synchronize();
            camera->film()->release();
        }
    }
};

luisa::unique_ptr<Integrator::Instance> AuxiliaryBufferPathTracing::build(
    Pipeline &pipeline, CommandBuffer &cmd_buffer) const noexcept {
    return luisa::make_unique<AuxiliaryBufferPathTracingInstance>(
        this, pipeline, cmd_buffer);
}

class AuxiliaryBuffer {

private:
    Pipeline &_pipeline;
    uint2 _resolution;
    uint _channels;
    Buffer<float> _buffer;

private:
    static constexpr auto clear_shader_name = luisa::string_view{"__aux_buffer_clear_shader"};

public:
    AuxiliaryBuffer(Pipeline &pipeline, uint2 resolution, uint channels, bool enabled = true) noexcept
        : _pipeline{pipeline}, _resolution{resolution},
          _channels{std::clamp(channels == 2u ? 3u : channels, 1u, 4u)} {
        _pipeline.register_shader<1u>(
            clear_shader_name, [](BufferFloat buffer) noexcept {
                buffer.write(dispatch_x(), 0.f);
            });
        if (enabled) {
            _buffer = pipeline.device().create_buffer<float>(
                _resolution.x * _resolution.y * _channels);
        }
    }
    void clear(CommandBuffer &command_buffer) const noexcept {
        if (_buffer) {
            command_buffer << _pipeline.shader<1u, Buffer<float>>(clear_shader_name, _buffer)
                                  .dispatch(_resolution.x * _resolution.y * _channels);
        }
    }
    [[nodiscard]] auto save(CommandBuffer &command_buffer,
                            std::filesystem::path path, uint total_samples) const noexcept
        -> luisa::function<void()> {
        if (!_buffer) { return {}; }
        auto host_image = luisa::make_shared<luisa::vector<float>>();
        host_image->resize(_resolution.x * _resolution.y * _channels);
        command_buffer << _buffer.copy_to(host_image->data());
        return [host_image, total_samples,
                resolution = _resolution,
                channels = _channels,
                path = std::move(path)] {
            auto scale = static_cast<float>(1. / total_samples);
            for (auto &p : *host_image) { p *= scale; }
            LUISA_INFO("Saving auxiliary buffer to '{}'.", path.string());
            save_image(path.string(), host_image->data(), resolution, channels);
        };
    }
    void accumulate(Expr<uint2> p, Expr<float4> value) noexcept {
        if (_buffer) {
            $if(!any(isnan(value))) {
                auto index = p.x + p.y * _resolution.x;
                for (auto i = 0u; i < _channels; i++) {
                    _buffer->atomic(index * _channels + i).fetch_add(value[i]);
                }
            };
        }
    }
};

void AuxiliaryBufferPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {

    auto spp = node<AuxiliaryBufferPathTracing>()->noisy_count();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto pixel_count = resolution.x * resolution.y;
    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    using namespace luisa::compute;

    // 3 diffuse, 3 specular, 3 normal, 1 depth, 3 albedo, 1 roughness, 1 emissive, 1 metallic, 1 transmissive, 1 specular-bounce
    luisa::unordered_map<luisa::string, luisa::unique_ptr<AuxiliaryBuffer>> aux_buffers;
    for (auto [comp, nc] : aov_component_to_channels) {
        auto enabled = node<AuxiliaryBufferPathTracing>()->is_component_enabled(comp);
        LUISA_INFO("Component {} is {}.", comp, enabled ? "enabled" : "disabled");
        auto v = luisa::make_unique<AuxiliaryBuffer>(pipeline(), resolution, nc, enabled);
        aux_buffers.emplace(comp, std::move(v));
    }

    // clear auxiliary buffers
    auto clear_auxiliary_buffers = [&] {
        for (auto &[_, buffer] : aux_buffers) {
            buffer->clear(command_buffer);
        }
    };

    Kernel2D render_auxiliary_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto camera_sample = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_sample.weight};
        SampledSpectrum Li{swl.dimension()};

        SampledSpectrum beta_diffuse{swl.dimension(), camera_sample.weight};
        SampledSpectrum Li_diffuse{swl.dimension()};

        auto ray = camera_sample.ray;
        auto pdf_bsdf = def(1e16f);
        auto specular_bounce = def(false);

        $for(depth, node<AuxiliaryBufferPathTracing>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            $if(depth == 0 & it->valid()) {
                aux_buffers.at("mask")->accumulate(dispatch_id().xy(), make_float4(1.f));
                aux_buffers.at("normal")->accumulate(dispatch_id().xy(), make_float4(it->shading().n(), 1.f));
                auto depth = length(it->p() - ray->origin());
                aux_buffers.at("depth")->accumulate(dispatch_id().xy(), make_float4(depth));
                auto p_ndc = make_float3((camera_sample.pixel / make_float2(resolution) * 2.f - 1.f) * make_float2(1.f, -1.f),
                                         depth / (ray->t_max() - ray->t_min()));
                aux_buffers.at("ndc")->accumulate(dispatch_id().xy(), make_float4(p_ndc, 1.f));
                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, 1.f, time);
                });
                call.execute([&](auto closure) noexcept {
                    auto albedo = closure->albedo();
                    auto roughness = closure->roughness();
                    aux_buffers.at("albedo")->accumulate(dispatch_id().xy(), make_float4(spectrum->srgb(swl, albedo), 1.f));
                    aux_buffers.at("roughness")->accumulate(dispatch_id().xy(), make_float4(roughness, 0.f, 1.f));
                });
            };

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    $if(!specular_bounce) {
                        Li_diffuse += beta_diffuse * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(
                        *it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    $if(!specular_bounce) {
                        Li_diffuse += beta_diffuse * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                };
            }

            $if(!it->shape().has_surface()) { $break; };

            // sample one light
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto eta_scale = def(1.f);

            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
                }

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

                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                        $if(!specular_bounce) {
                            Li_diffuse += w * beta_diffuse * eval.f * light_sample.eval.L;
                        };
                    };

                    // sample material
                    auto sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(sample.wi);
                    pdf_bsdf = sample.eval.pdf;
                    auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                    beta *= w * sample.eval.f;
                    $if(!specular_bounce) {
                        beta_diffuse *= w * sample.eval.f;
                    };

                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                };
                specular_bounce = all(closure->roughness() < .05f);
            });
        };
        aux_buffers.at("sample")->accumulate(pixel_id, make_float4(spectrum->srgb(swl, Li * shutter_weight), 1.f));
        aux_buffers.at("diffuse")->accumulate(pixel_id, make_float4(spectrum->srgb(swl, Li_diffuse * shutter_weight), 1.f));
        aux_buffers.at("specular")->accumulate(pixel_id, make_float4(spectrum->srgb(swl, (Li - Li_diffuse) * shutter_weight), 1.f));
    };

    Clock clock_compile;
    auto render_auxiliary = pipeline().device().compile(render_auxiliary_kernel);
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    Clock clock;
    ProgressBar progress;
    progress.update(0.0);

    auto dispatch_count = 0u;
    auto dispatches_per_commit = 32u;
    auto aux_spp = node<AuxiliaryBufferPathTracing>()->noisy_count();
    auto should_dump = [this, aux_spp](uint32_t n) -> bool {
        auto strategy = node<AuxiliaryBufferPathTracing>()->dump_strategy();
        if (strategy == AuxiliaryBufferPathTracing::DumpStrategy::POWER2) {
            return n > 0 && ((n & (n - 1)) == 0);
        }
        if (strategy == AuxiliaryBufferPathTracing::DumpStrategy::ALL) {
            return true;
        }
        return n == aux_spp;
    };
    LUISA_ASSERT(shutter_samples.size() == 1u || camera->node()->spp() == aux_spp,
                 "AOVIntegrator is not compatible with motion blur "
                 "if rendered with different spp from the camera.");
    if (aux_spp != camera->node()->spp()) {
        Camera::ShutterSample ss{
            .point = {.time = camera->node()->shutter_span().x,
                      .weight = 1.f},
            .spp = aux_spp};
        shutter_samples = {ss};
    }
    auto sample_count = 0u;
    for (auto s : shutter_samples) {
        pipeline().update(command_buffer, s.point.time);
        clear_auxiliary_buffers();
        auto parent_path = camera->node()->file().parent_path();
        auto filename = camera->node()->file().stem().string();
        auto ext = camera->node()->file().extension().string();
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render_auxiliary(sample_count++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (should_dump(sample_count)) {
                LUISA_INFO("Saving AOVs at sample #{}.", sample_count);
                luisa::vector<luisa::function<void()>> savers;
                for (auto &[component, buffer] : aux_buffers) {
                    auto path = node<AuxiliaryBufferPathTracing>()->dump_strategy() ==
                                        AuxiliaryBufferPathTracing::DumpStrategy::FINAL ?
                                    parent_path / fmt::format("{}_{}{}", filename, component, ext) :
                                    parent_path / fmt::format("{}_{}_{:05}{}", filename, component, sample_count, ext);
                    if (auto saver = buffer->save(command_buffer, path, sample_count)) {
                        savers.emplace_back(std::move(saver));
                    }
                }
                if (!savers.empty()) {
                    command_buffer << [&] { for (auto &s : savers) { s(); } }
                                   << synchronize();
                }
            }
            if (sample_count % 16u == 0u) { command_buffer << commit(); }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::AuxiliaryBufferPathTracing)
