//
// Created by Mike Smith on 2022/1/10.
//

#include <fstream>

#include <core/json.h>
#include <ast/function_serializer.h>
#include <util/imageio.h>
#include <luisa-compute.h>

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
                              {"ndc", 3u}};

class AuxiliaryBufferPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _noisy_count;
    luisa::unordered_set<luisa::string_view> _enabled_aov;

public:
    AuxiliaryBufferPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _noisy_count{std::max(desc->property_uint_or_default("noisy_count", 8u), 8u)} {
        for (auto [c, _] : aov_component_to_channels) {
            auto option = luisa::format("enable_{}", c);
            if (desc->property_bool_or_default(option, true)) {
                _enabled_aov.emplace(c);
            }
        }
    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto noisy_count() const noexcept { return _noisy_count; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return false; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
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
    void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline,
        AuxiliaryBufferPathTracingInstance *pt, Camera::Instance *camera) noexcept;

public:
    explicit AuxiliaryBufferPathTracingInstance(
        const AuxiliaryBufferPathTracing *node,
        Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<AuxiliaryBufferPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, pipeline(), this, camera);
            command_buffer << compute::synchronize();
            camera->film()->release();
        }
        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
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
    Image<float> _image;

private:
    static constexpr auto clear_shader_name = luisa::string_view{"__aux_buffer_clear_shader"};

public:
    AuxiliaryBuffer(Pipeline &pipeline, uint2 resolution, uint channels, bool enabled = true) noexcept
        : _pipeline{pipeline} {
        _pipeline.register_shader<2u>(
            clear_shader_name, [](ImageFloat image) noexcept {
                image.write(dispatch_id().xy(), make_float4(0.f));
            });
        if (enabled) {
            _image = pipeline.device().create_image<float>(
                channels == 1u ?// TODO: support FLOAT2
                    PixelStorage::FLOAT1 :
                    PixelStorage::FLOAT4,
                resolution);
        }
    }
    void clear(CommandBuffer &command_buffer) const noexcept {
        if (_image) {
            command_buffer << _pipeline.shader<2u, Image<float>>(clear_shader_name, _image)
                                  .dispatch(_image.size());
        }
    }
    [[nodiscard]] auto save(CommandBuffer &command_buffer,
                            std::filesystem::path path, uint total_samples) const noexcept
        -> luisa::function<void()> {
        if (!_image) { return {}; }
        auto host_image = luisa::make_shared<luisa::vector<float>>();
        auto nc = pixel_storage_channel_count(_image.storage());
        host_image->resize(_image.size().x * _image.size().y * nc);
        command_buffer << _image.copy_to(host_image->data());
        return [host_image, total_samples, nc, size = _image.size(), path = std::move(path)] {
            auto scale = static_cast<float>(1. / total_samples);
            for (auto &p : *host_image) { p *= scale; }
            LUISA_INFO("Saving auxiliary buffer to '{}'.", path.string());
            save_image(path.string(), host_image->data(), size, nc);
        };
    }
    void accumulate(Expr<uint2> p, Expr<float4> value) noexcept {
        if (_image) {
            $if(!any(isnan(value))) {
                auto old = _image.read(p);
                _image.write(p, old + value);
            };
        }
    }
};

void AuxiliaryBufferPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline,
    AuxiliaryBufferPathTracingInstance *pt, Camera::Instance *camera) noexcept {

    auto spp = node<AuxiliaryBufferPathTracing>()->noisy_count();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    if (!pipeline.has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto light_sampler = pt->light_sampler();
    auto sampler = pt->sampler();
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;

    // 3 diffuse, 3 specular, 3 normal, 1 depth, 3 albedo, 1 roughness, 1 emissive, 1 metallic, 1 transmissive, 1 specular-bounce
    luisa::unordered_map<luisa::string, luisa::unique_ptr<AuxiliaryBuffer>> aux_buffers;
    for (auto [comp, nc] : aov_component_to_channels) {
        auto enabled = node<AuxiliaryBufferPathTracing>()->is_component_enabled(comp);
        auto v = luisa::make_unique<AuxiliaryBuffer>(pipeline, resolution, nc, enabled);
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
        sampler->start(pixel_id, frame_index);
        auto camera_sample = camera->generate_ray(*sampler, pixel_id, time);
        auto spectrum = pipeline.spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_sample.weight};
        SampledSpectrum Li{swl.dimension()};

        SampledSpectrum beta_diffuse{swl.dimension(), camera_sample.weight};
        SampledSpectrum Li_diffuse{swl.dimension()};

        auto ray = camera_sample.ray;
        auto pdf_bsdf = def(1e16f);
        auto specular_bounce = def(false);

        $for(depth, pt->node<AuxiliaryBufferPathTracing>()->max_depth()) {

            // trace
            auto it = pipeline.geometry()->intersect(ray);

            $if(depth == 0 & it->valid()) {
                aux_buffers.at("normal")->accumulate(dispatch_id().xy(), make_float4(it->shading().n(), 1.f));
                aux_buffers.at("depth")->accumulate(dispatch_id().xy(), make_float4(length(it->p() - ray->origin())));
                auto p_cs = make_float3(inverse(camera->camera_to_world()) * make_float4(it->p(), 1.f));
                auto clip = camera->node()->clip_plane();
                auto p_ndc = make_float3((camera_sample.pixel / make_float2(resolution) * 2.f - 1.f) * make_float2(1.f, -1.f),
                                         (-p_cs.z - clip.x) / (clip.y - clip.x));
                aux_buffers.at("ndc")->accumulate(dispatch_id().xy(), make_float4(p_ndc, 1.f));
                pipeline.surfaces().dispatch(it->shape()->surface_tag(), [&](auto surface) noexcept {
                    // create closure
                    auto closure = surface->closure(*it, swl, 1.f, time);
                    auto albedo = closure->albedo();
                    auto roughness = closure->roughness();
                    aux_buffers.at("albedo")->accumulate(dispatch_id().xy(), make_float4(spectrum->srgb(swl, albedo), 1.f));
                    aux_buffers.at("roughness")->accumulate(dispatch_id().xy(), make_float4(roughness, 0.f, 1.f));
                });
            };

            // miss
            $if(!it->valid()) {
                if (pipeline.environment()) {
                    auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    $if(!specular_bounce) {
                        Li_diffuse += beta_diffuse * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                }
                $break;
            };

            // hit light
            if (!pipeline.lights().empty()) {
                $if(it->shape()->has_light()) {
                    auto eval = light_sampler->evaluate_hit(
                        *it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    $if(!specular_bounce) {
                        Li_diffuse += beta_diffuse * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                };
            }

            $if(!it->shape()->has_surface()) { $break; };

            // sample one light
            auto u_light_selection = sampler->generate_1d();
            auto u_light_surface = sampler->generate_2d();
            Light::Sample light_sample = light_sampler->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
            auto occluded = pipeline.geometry()->intersect_any(shadow_ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler->generate_1d();
            auto u_bsdf = sampler->generate_2d();
            auto eta_scale = def(1.f);

            pipeline.surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
                }

                // apply roughness map
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
                    auto wo = -ray->direction();

                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.wi;
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
            });

            pipeline.surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                auto closure = surface->closure(*it, swl, 1.f, time);
                specular_bounce = false;
                $if((closure->roughness().x < 0.05f) & (closure->roughness().y < 0.05f)) {
                    specular_bounce = true;
                };
            });
            // rr is closed for aov
        };
        aux_buffers.at("sample")->accumulate(pixel_id, make_float4(spectrum->srgb(swl, Li * shutter_weight), 1.f));
        aux_buffers.at("diffuse")->accumulate(pixel_id, make_float4(spectrum->srgb(swl, Li_diffuse * shutter_weight), 1.f));
        aux_buffers.at("specular")->accumulate(pixel_id, make_float4(spectrum->srgb(swl, (Li - Li_diffuse) * shutter_weight), 1.f));
    };

    Clock clock_compile;
    auto render_auxiliary = pipeline.device().compile(render_auxiliary_kernel);
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
    auto aux_spp = pt->node<AuxiliaryBufferPathTracing>()->noisy_count();
    auto check_sample_output = [](uint32_t n) -> bool {
        return n > 0 && ((n & (n - 1)) == 0);
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
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline.update(command_buffer, s.point.time);
        clear_auxiliary_buffers();
        auto parent_path = camera->node()->file().parent_path();
        auto filename = camera->node()->file().stem().string();
        auto ext = camera->node()->file().extension().string();
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render_auxiliary(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            luisa::vector<luisa::function<void()>> savers;
            if (check_sample_output(sample_id)) {
                for (auto &[component, buffer] : aux_buffers) {
                    auto path = parent_path / fmt::format("{}_{}{:05}{}", filename, component, sample_id, ext);
                    if (auto saver = buffer->save(command_buffer, path, sample_id)) {
                        savers.emplace_back(std::move(saver));
                    }
                }
                if (!savers.empty()) {
                    command_buffer << [&] { for (auto &s : savers) { s(); } }
                                   << synchronize();
                }
            }
            if (++sample_id % 16u == 0u) { command_buffer << commit(); }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
    {
        std::ofstream file{"results.txt", std::ios::app};
        file << "Render time = " << render_time << " ms" << std::endl;
    }
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::AuxiliaryBufferPathTracing)
