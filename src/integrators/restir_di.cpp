//
// Created by Leon Kang on 2024/4/11.
//

//
// Created by Mike Smith on 2022/1/10.
//

#include <util/progress_bar.h>
#include <util/imageio.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class ReSTIRDirectLighting final : public ProgressiveIntegrator {

private:
    uint _num_initial_sample;
    uint _num_neighbor_sample;
    bool _enable_temporal_reuse;
    bool _enable_spatial_reuse;
    bool _enable_visibility_reuse;

public:
    ReSTIRDirectLighting(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _num_initial_sample{desc->property_uint_or_default("num_initial_sample", 1u)},
          _num_neighbor_sample{desc->property_uint_or_default("num_neighbor_sample", 5u)},
          _enable_temporal_reuse{desc->property_bool_or_default("enable_temporal_reuse", true)},
          _enable_spatial_reuse{desc->property_bool_or_default("enable_spatial_reuse", true)},
          _enable_visibility_reuse{desc->property_bool_or_default("enable_visibility_reuse", true)} {}
    [[nodiscard]] auto num_initial_sample() const noexcept { return _num_initial_sample; }
    [[nodiscard]] auto num_neighbor_sample() const noexcept { return _num_neighbor_sample; }
    [[nodiscard]] auto enable_spatial_reuse() const noexcept { return _enable_spatial_reuse; }
    [[nodiscard]] auto enable_visibility_reuse() const noexcept { return _enable_visibility_reuse; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ReSTIRDirectLightingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

    struct ReservoirSample {
        UInt tag;
        Float2 u_light_selection;
    };
    struct ReservoirWeight {
        Float m;
        Float total_weight;
        Float target_pdf;
    };
    struct Reservoir {
        ReservoirSample sample;
        ReservoirWeight weight;
        [[nodiscard]] static auto zero() noexcept {
            return Reservoir{ReservoirSample{0u, make_float2(0.f)}, ReservoirWeight{0.f, 0.f, 0.f}};
        }
        [[nodiscard]] auto contribution_weight() const noexcept {
            return ite(weight.total_weight == 0.f, 0.f, weight.total_weight / weight.target_pdf);
        }
        void update(const Reservoir &r, Expr<float> u_sel) noexcept {
            auto w1 = weight.total_weight * weight.m,
                 w2 = r.weight.total_weight * r.weight.m;
            weight.m = weight.m + r.weight.m;
            weight.total_weight = w1 + w2;
            $if (u_sel * weight.total_weight < w2) {
                sample = r.sample;
                weight.target_pdf = r.weight.target_pdf;
            };
            weight.total_weight /= weight.m;
        }
        void update(const Reservoir &r, Expr<float> m1, Expr<float> m2, Expr<float> u_sel) noexcept {
            auto w1 = weight.total_weight * m1, w2 = r.weight.total_weight * m2;
            weight.m = weight.m + r.weight.m;
            weight.total_weight = w1 + w2;
            $if (u_sel * weight.total_weight < w2) {
                sample = r.sample;
                weight.target_pdf = r.weight.target_pdf;
            };
        }
    };
private:
    [[nodiscard]] std::pair<SampledSpectrum, Float> _evaluate_reservoir_sample(const ReservoirSample &sample, const Interaction &it, Expr<float3> wo,
                                                                               const SampledWavelengths &swl, Expr<float> time, Expr<bool> occlusion_test = false) const noexcept {
        auto L = SampledSpectrum{swl.dimension(), 0.f};
        auto pdf = def(0.f);
        auto prob = light_sampler()->evaluate_selection(sample.tag, it.p(), swl, time);
        auto sel = LightSampler::Selection{sample.tag, prob};
        auto light_sample = light_sampler()->sample_light(it, sel, sample.u_light_selection, swl, time);
        auto occluded = def(false);
        $if(occlusion_test) {
            occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
        };
        $if(light_sample.eval.pdf > 0.f & !occluded) {
            auto surface_tag = it.shape().surface_tag();
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                auto wi = light_sample.shadow_ray->direction();
                auto eval = closure->evaluate(wo, wi);
                pdf = light_sample.eval.pdf;
                L = eval.f * light_sample.eval.L;
            });
        };
        return std::make_pair(L, pdf);
    }
    [[nodiscard]] Reservoir _generate_reservoir(const Interaction &it, Expr<float3> wo, const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto u_sel = sampler()->generate_1d();
        auto sel = light_sampler()->select(it, u_sel, swl, time);
        auto u_light_selection = sampler()->generate_2d();
        auto reservoir_sample = ReservoirSample{sel.tag, u_light_selection};
        auto [L, pdf] = _evaluate_reservoir_sample(reservoir_sample, it, wo, swl, time);
        auto target_pdf = L.sum();
        auto total_weight = ite(pdf == 0.f, 0.f, target_pdf / pdf);
        return Reservoir{reservoir_sample, ReservoirWeight{1.f, total_weight, target_pdf}};
    }
public:
    class ReservoirBuffer {
    private:
        uint2 _resolution;
        Buffer<uint3> _sample;
        Buffer<float3> _weight;
    public:
        ReservoirBuffer(const Pipeline &pipeline, uint2 resolution) noexcept
            : _resolution{resolution} {
            auto num_pixel = resolution.x * resolution.y;
            _sample = pipeline.device().create_buffer<uint3>(num_pixel);
            _weight = pipeline.device().create_buffer<float3>(num_pixel);
        }
        [[nodiscard]] auto read(Expr<uint2> pixel_id) const noexcept {
            auto sample = _sample->read(pixel_id.x + pixel_id.y * _resolution.x);
            auto weight = _weight->read(pixel_id.x + pixel_id.y * _resolution.x);
            auto reservoir_sample = ReservoirSample{sample.x, weight.xy().as<float2>()};
            auto reservoir_weight = ReservoirWeight{weight.z, weight.y, weight.x};
            return Reservoir{reservoir_sample, reservoir_weight};
        }
        void write(const Reservoir &r, Expr<uint2> pixel_id) noexcept {
            auto sample = make_uint3(r.sample.tag, r.sample.u_light_selection.as<uint2>());
            auto weight = make_float3(r.weight.m, r.weight.total_weight, r.weight.target_pdf);
            _sample->write(pixel_id.x + pixel_id.y * _resolution.x, sample);
            _weight->write(pixel_id.x + pixel_id.y * _resolution.x, weight);
        }
    };
protected:
    void _render_one_camera(CommandBuffer &command_buffer,
                            Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();

        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        command_buffer << pipeline().printer().reset();
        command_buffer << compute::synchronize();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);
        auto spatial_reservoir_buffer = luisa::make_unique<ReservoirBuffer>(pipeline(), resolution);
        using namespace luisa::compute;
        // Kernel2D temporal_reuse_kernel
        // Kernel2D spatial_reuse_kernel
        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        };

        Clock clock_compile;
        auto render = pipeline().device().compile(render_kernel);
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        luisa::vector<float4> local_pixels;
        if (node()->video()) {
            shutter_samples = camera->node()->uniform_shutter_samples();
            local_pixels.resize(pixel_count);
        }
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;

        auto shutter_id = 0u;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            for (auto i = 0u; i < s.spp; i++) {
                command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                      .dispatch(resolution);
                if (auto &&p = pipeline().printer(); !p.empty()) {
                    command_buffer << p.retrieve();
                }
                dispatch_count++;
                if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
                auto dispatches_per_commit = 4u;
                if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
            if (node()->video()) {
                command_buffer << synchronize();
                camera->film()->download(command_buffer, local_pixels.data());
                command_buffer << compute::synchronize();
                camera->film()->clear(command_buffer);
                if (node()->save()) {
                    auto film_path = camera->node()->file();
                    //film_path is a std::filesystem::path, add number to its name
                    auto new_name = film_path.stem().string() + std::format("{:05}", shutter_id) + film_path.extension().string();
                    auto new_film_path = film_path.replace_filename(new_name);
                    save_image(new_film_path, reinterpret_cast<const float *>(local_pixels.data()), resolution);
                }
                shutter_id++;
            }
        }
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};

        auto ray = cs.ray;

        $loop {
            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += cs.weight * eval.L;
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += cs.weight * eval.L;
                };
            }

            // compute direct lighting
            $if(!it->shape().has_surface()) { $break; };
            auto reservoir = Reservoir::zero();
            $for(_, node<ReSTIRDirectLighting>()->num_initial_sample()) {
                auto candidate = _generate_reservoir(*it, wo, swl, time);
                reservoir.update(candidate, sampler()->generate_1d());
            };
            auto [L, _] = _evaluate_reservoir_sample(reservoir.sample, *it, wo, swl, time, true);
            auto contribution_weight = reservoir.contribution_weight();
            Li += cs.weight * contribution_weight * L;
            $break;
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> ReSTIRDirectLighting::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ReSTIRDirectLightingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ReSTIRDirectLighting)