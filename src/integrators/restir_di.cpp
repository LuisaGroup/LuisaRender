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
    bool _enable_visibility_reuse;
    bool _enable_temporal_reuse;
    bool _enable_decorrelation;
    bool _enable_spatial_reuse;

public:
    ReSTIRDirectLighting(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _num_initial_sample{desc->property_uint_or_default("num_initial_sample", 1u)},
          _enable_visibility_reuse{desc->property_bool_or_default("enable_visibility_reuse", true)},
          _enable_temporal_reuse{desc->property_bool_or_default("enable_temporal_reuse", true)},
          _enable_decorrelation{desc->property_bool_or_default("enable_decorrelation", true)},
          _enable_spatial_reuse{desc->property_bool_or_default("enable_spatial_reuse", true)} {}
    [[nodiscard]] auto num_initial_sample() const noexcept { return _num_initial_sample; }
    [[nodiscard]] auto enable_visibility_reuse() const noexcept { return _enable_visibility_reuse; }
    [[nodiscard]] auto enable_temporal_reuse() const noexcept { return _enable_temporal_reuse; }
    [[nodiscard]] auto enable_decorrelation() const noexcept { return _enable_decorrelation; }
    [[nodiscard]] auto enable_spatial_reuse() const noexcept { return _enable_spatial_reuse; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ReSTIRDirectLightingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

    struct ReservoirSample {
        UInt tag;
        Float2 u_light_surface;
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
            $if(u_sel * weight.total_weight < w2) {
                sample = r.sample;
                weight.target_pdf = r.weight.target_pdf;
            };
            weight.total_weight /= weight.m;
        }
        void update(const Reservoir &r, Expr<float> m1, Expr<float> m2, Expr<float> u_sel) noexcept {
            auto w1 = weight.total_weight * m1, w2 = r.weight.total_weight * m2;
            weight.m = weight.m + r.weight.m;
            weight.total_weight = w1 + w2;
            $if(u_sel * weight.total_weight < w2) {
                sample = r.sample;
                weight.target_pdf = r.weight.target_pdf;
            };
        }
        void accept(const Reservoir &r, Expr<float> u_sel) noexcept {
            $if(r.weight.total_weight > weight.total_weight | u_sel < r.weight.total_weight / weight.total_weight) {
                sample = r.sample;
                weight = r.weight;
            };
        }
    };

private:
    [[nodiscard]] std::pair<SampledSpectrum, Float> _evaluate_without_occlusion(const ReservoirSample &sample, const Interaction &it, Expr<float3> wo,
                                                                                const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto L = SampledSpectrum{swl.dimension(), 0.f};
        auto pdf = def(0.f);
        auto prob = light_sampler()->evaluate_selection(sample.tag, it.p(), swl, time);
        auto sel = LightSampler::Selection{sample.tag, prob};
        auto light_sample = light_sampler()->sample_light(it, sel, sample.u_light_surface, swl, time);
        auto surface_tag = it.shape().surface_tag();
        PolymorphicCall<Surface::Closure> call;
        pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
            surface->closure(call, it, swl, wo, 1.f, time);
        });
        call.execute([&](auto closure) noexcept {
            $if(light_sample.eval.pdf > 0.f) {
                auto wi = light_sample.shadow_ray->direction();
                auto eval = closure->evaluate(wo, wi);
                pdf = light_sample.eval.pdf;
                L = eval.f * light_sample.eval.L;
            };
        });
        return std::make_pair(L, pdf);
    }
    [[nodiscard]] std::pair<SampledSpectrum, Float> _evaluate_with_occlusion(const ReservoirSample &sample, const Interaction &it, Expr<float3> wo,
                                                                             const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto L = SampledSpectrum{swl.dimension(), 0.f};
        auto pdf = def(0.f);
        auto prob = light_sampler()->evaluate_selection(sample.tag, it.p(), swl, time);
        auto sel = LightSampler::Selection{sample.tag, prob};
        auto light_sample = light_sampler()->sample_light(it, sel, sample.u_light_surface, swl, time);
        auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
        auto surface_tag = it.shape().surface_tag();
        PolymorphicCall<Surface::Closure> call;
        pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
            surface->closure(call, it, swl, wo, 1.f, time);
        });
        call.execute([&](auto closure) noexcept {
            $if(light_sample.eval.pdf > 0.f & !occluded) {
                auto wi = light_sample.shadow_ray->direction();
                auto eval = closure->evaluate(wo, wi);
                pdf = light_sample.eval.pdf;
                L = eval.f * light_sample.eval.L;
            };
        });
        return std::make_pair(L, pdf);
    }

public:
    class VisibilityBuffer {
    private:
        uint2 _resolution;
        Buffer<float> _weight;
        Buffer<Ray> _ray;
        Buffer<Hit> _hit;

    public:
        VisibilityBuffer(const Pipeline &pipeline, uint2 resolution) noexcept
            : _resolution{resolution} {
            auto num_pixel = resolution.x * resolution.y;
            _ray = pipeline.device().create_buffer<Ray>(num_pixel);
            _hit = pipeline.device().create_buffer<Hit>(num_pixel);
            _weight = pipeline.device().create_buffer<float>(num_pixel);
        }
        void set_weight(Expr<float> weight, Expr<uint2> pixel_id) noexcept {
            _weight->write(pixel_id.x + pixel_id.y * _resolution.x, weight);
        }
        [[nodiscard]] auto weight(Expr<uint2> pixel_id) const noexcept {
            return _weight->read(pixel_id.x + pixel_id.y * _resolution.x);
        }
        void set_ray(Expr<Ray> ray, Expr<uint2> pixel_id) noexcept {
            _ray->write(pixel_id.x + pixel_id.y * _resolution.x, ray);
        }
        [[nodiscard]] auto ray(Expr<uint2> pixel_id) const noexcept {
            return _ray->read(pixel_id.x + pixel_id.y * _resolution.x);
        }
        void set_hit(Expr<Hit> hit, Expr<uint2> pixel_id) noexcept {
            _hit->write(pixel_id.x + pixel_id.y * _resolution.x, hit);
        }
        [[nodiscard]] auto hit(Expr<uint2> pixel_id) const noexcept {
            return _hit->read(pixel_id.x + pixel_id.y * _resolution.x);
        }
    };
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
            auto reservoir_sample = ReservoirSample{sample.x, sample.yz().as<float2>()};
            auto reservoir_weight = ReservoirWeight{weight.x, weight.y, weight.z};
            return Reservoir{reservoir_sample, reservoir_weight};
        }
        void write(const Reservoir &r, Expr<uint2> pixel_id) noexcept {
            auto sample = make_uint3(r.sample.tag, r.sample.u_light_surface.as<uint2>());
            auto weight = make_float3(r.weight.m, r.weight.total_weight, r.weight.target_pdf);
            _sample->write(pixel_id.x + pixel_id.y * _resolution.x, sample);
            _weight->write(pixel_id.x + pixel_id.y * _resolution.x, weight);
        }
    };

private:
    luisa::unique_ptr<ReservoirBuffer> _spatial_reservoir_buffer;
    luisa::unique_ptr<ReservoirBuffer> _temporal_reservoir_buffer;
    luisa::unique_ptr<VisibilityBuffer> _visibility_buffer;
    uint _total_frame_count{0u};
    Buffer<float4x4> _prev_frame_view_matrix;
    void _generate_sample(Expr<uint> num_initial_sample, const Camera::Instance *camera, Expr<uint> frame_index,
                          Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        _visibility_buffer->set_weight(cs.weight, pixel_id);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = cs.ray;
        _visibility_buffer->set_ray(ray, pixel_id);
        auto hit = pipeline().geometry()->trace_closest(ray);
        _visibility_buffer->set_hit(hit, pixel_id);
        auto reservoir = Reservoir::zero();
        $loop {
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->interaction(ray, hit);
            // miss
            $if(!it->valid() | !it->shape().has_surface()) { $break; };
            // RIS over light samples
            $outline {
                $for(_, num_initial_sample) {
                    auto u_sel = sampler()->generate_1d();
                    auto sel = light_sampler()->select(*it, u_sel, swl, time);
                    auto u_light_selection = sampler()->generate_2d();
                    auto target_pdf = def(0.f), total_weight = def(0.f);
                    Reservoir candidate{
                        ReservoirSample{sel.tag, u_light_selection},
                        ReservoirWeight{1.f, total_weight, target_pdf}};
                    auto [L, pdf] = _evaluate_without_occlusion(candidate.sample, *it, wo, swl, time);
                    candidate.weight.target_pdf = L.sum();
                    candidate.weight.total_weight = ite(pdf == 0.f, 0.f, candidate.weight.target_pdf / pdf);
                    reservoir.update(candidate, sampler()->generate_1d());
                };
            };
            $break;
        };
        $if(dsl::isnan(reservoir.weight.total_weight) | dsl::isnan(reservoir.weight.target_pdf)) {
            reservoir = Reservoir::zero();
        };
        _spatial_reservoir_buffer->write(reservoir, pixel_id);
    }
    void _visibility_reuse(Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->ray(pixel_id);
        auto hit = _visibility_buffer->hit(pixel_id);
        auto reservoir = _spatial_reservoir_buffer->read(pixel_id);
        $loop {
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->interaction(ray, hit);
            // miss
            $if(!it->valid() | !it->shape().has_surface()) { $break; };
            // visibility reuse
            $outline {
                auto light_sample = light_sampler()->sample_light(*it, {reservoir.sample.tag, 1.f}, reservoir.sample.u_light_surface, swl, time);
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                $if(light_sample.eval.pdf == 0.f | occluded) {
                    reservoir.weight.total_weight = 0.f;
                };
            };
            $break;
        };
        _spatial_reservoir_buffer->write(reservoir, pixel_id);
    }
    void _temporal_reuse(const Camera::Instance *camera, Expr<uint> frame_index,
                         Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        auto resolution = camera->film()->node()->resolution();
        sampler()->start(pixel_id, frame_index);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->ray(pixel_id);
        auto hit = _visibility_buffer->hit(pixel_id);
        auto reservoir = _spatial_reservoir_buffer->read(pixel_id);
        $loop {
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->interaction(ray, hit);
            // miss
            $if(!it->valid() | !it->shape().has_surface()) { $break; };
            // temporal reuse
            $if(frame_index == 0u) { $break; };
            auto prev_frame_view_matrix = _prev_frame_view_matrix->read(0u);
            auto p_view = make_float3(prev_frame_view_matrix * make_float4(it->p(), 1.f));
            auto [prev_frame_pixel_id, valid] = camera->project(p_view);
            prev_frame_pixel_id = clamp(prev_frame_pixel_id, 0.f, make_float2(resolution) - 1.f);
            $if(valid) {
                auto prev_frame_reservoir = _temporal_reservoir_buffer->read(make_uint2(prev_frame_pixel_id));
                $if(dsl::isnan(prev_frame_reservoir.weight.total_weight) | dsl::isnan(prev_frame_reservoir.weight.target_pdf)) { $break; };
                $if(prev_frame_reservoir.weight.target_pdf > 0.f) {
                    auto [L, pdf] = _evaluate_without_occlusion(prev_frame_reservoir.sample, *it, wo, swl, time);
                    auto target_pdf = L.sum();
                    prev_frame_reservoir.weight.total_weight *= target_pdf / prev_frame_reservoir.weight.target_pdf;
                    prev_frame_reservoir.weight.target_pdf = target_pdf;
                    prev_frame_reservoir.weight.m = min(prev_frame_reservoir.weight.m, 20.f * reservoir.weight.m);
                    reservoir.update(prev_frame_reservoir, sampler()->generate_1d());
                };
            };
            $break;
        };
        _spatial_reservoir_buffer->write(reservoir, pixel_id);
    }
    void _spatial_reuse(Expr<uint> pass_index, const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        auto resolution = camera->film()->node()->resolution();
        sampler()->start(pixel_id << 1u | pass_index, frame_index);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->ray(pixel_id);
        auto hit = _visibility_buffer->hit(pixel_id);
        auto constexpr num_neighbor_sample = 5u;
        auto constexpr neighbor_radius = 30.f;
        auto reservoir = Reservoir::zero();
        $loop {
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->interaction(ray, hit);
            // miss
            $if(!it->valid() | !it->shape().has_surface()) { $break; };
            // spatial reuse
            $outline {
                $if(pass_index % 2u == 0u) {
                    reservoir = _spatial_reservoir_buffer->read(pixel_id);
                }
                $else {
                    reservoir = _temporal_reservoir_buffer->read(pixel_id);
                };
                auto camera_to_world = camera->camera_to_world();
                auto world_to_camera = inverse(camera_to_world);
                auto current_pixel_depth = (world_to_camera * make_float4(it->p(), 1.f)).z;
                $for(_, num_neighbor_sample) {
                    auto u_radius = sampler()->generate_1d(), u_theta = sampler()->generate_1d();
                    auto radius = neighbor_radius * sqrt(u_radius);
                    auto theta = 2.f * pi * u_theta;
                    auto offset = make_float2(radius * cos(theta), radius * sin(theta));
                    auto neighbor_id = make_uint2(clamp(make_float2(pixel_id) + offset, make_float2(0.f), make_float2(resolution) - 1.f));
                    auto neighbor_ray = _visibility_buffer->ray(neighbor_id);
                    auto neighbor_hit = _visibility_buffer->hit(neighbor_id);
                    auto neighbor_it = pipeline().geometry()->interaction(neighbor_ray, neighbor_hit);
                    $if(neighbor_it->valid() & neighbor_it->shape().has_surface()) {
                        auto neighbor_pixel_depth = (world_to_camera * make_float4(neighbor_it->p(), 1.f)).z;
                        $if(abs(neighbor_pixel_depth - current_pixel_depth) < 0.1f * abs(current_pixel_depth) &
                            dot(it->ng(), neighbor_it->ng()) > 0.91f) {
                            auto neighbor_reservoir = Reservoir::zero();
                            $if(pass_index % 2u == 0u) {
                                neighbor_reservoir = _spatial_reservoir_buffer->read(neighbor_id);
                            }
                            $else {
                                neighbor_reservoir = _temporal_reservoir_buffer->read(neighbor_id);
                            };
                            $if(dsl::isnan(neighbor_reservoir.weight.total_weight) | dsl::isnan(neighbor_reservoir.weight.target_pdf)) { $continue; };
                            auto [L, pdf] = _evaluate_without_occlusion(neighbor_reservoir.sample, *it, wo, swl, time);
                            $if(neighbor_reservoir.weight.target_pdf > 0.f) {
                                auto neighbor_target_pdf = L.sum();
                                neighbor_reservoir.weight.total_weight *= neighbor_target_pdf / neighbor_reservoir.weight.target_pdf;
                                neighbor_reservoir.weight.target_pdf = neighbor_target_pdf;
                                reservoir.update(neighbor_reservoir, sampler()->generate_1d());
                            };
                        };
                    };
                };
            };
            $break;
        };
        $if(pass_index % 2u == 0u) {
            _temporal_reservoir_buffer->write(reservoir, pixel_id);
        }
        $else {
            _spatial_reservoir_buffer->write(reservoir, pixel_id);
        };
    }
    void _perturb_sample(Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept {
        auto constexpr num_perturb_iter = 10u;
        auto constexpr sample_gaussian = [](Expr<float2> u) noexcept {
            Float2 uu = u;
            uu.x = 2.f * abs(u.x - .5f);
            auto x = ite(u.x < .5f, -1.f, 1.f) * sqrt(-log(uu));
            $if(any(dsl::isnan(x))) {
                x = make_float2(0.f);
            };
            return x;
        };
        sampler()->start(pixel_id, frame_index);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto ray = _visibility_buffer->ray(pixel_id);
        auto hit = _visibility_buffer->hit(pixel_id);
        auto reservoir = _spatial_reservoir_buffer->read(pixel_id);
        $loop {
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->interaction(ray, hit);
            // miss
            $if(!it->valid() | !it->shape().has_surface()) { $break; };
            // offset the sample location on the light surface
            $for(_, num_perturb_iter) {
                auto candidate = reservoir;
                auto perturbation = 0.01f * sample_gaussian(sampler()->generate_2d());
                candidate.sample.u_light_surface = clamp(reservoir.sample.u_light_surface + perturbation, 0.f, 1.f);
                auto [L, pdf] = _evaluate_without_occlusion(candidate.sample, *it, wo, swl, time);
                $if(reservoir.weight.target_pdf > 0.f & pdf > 0.f) {
                    auto target_pdf = L.sum();
                    candidate.weight.total_weight *= target_pdf / candidate.weight.target_pdf;
                    candidate.weight.target_pdf = target_pdf;
                    reservoir.accept(candidate, sampler()->generate_1d());
                };
            };
            $break;
        };
        _spatial_reservoir_buffer->write(reservoir, pixel_id);
    }

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
        if (!_spatial_reservoir_buffer) {
            _spatial_reservoir_buffer = luisa::make_unique<ReservoirBuffer>(pipeline(), resolution);
        }
        if (!_temporal_reservoir_buffer) {
            _temporal_reservoir_buffer = luisa::make_unique<ReservoirBuffer>(pipeline(), resolution);
        }
        if (!_visibility_buffer) {
            _visibility_buffer = luisa::make_unique<VisibilityBuffer>(pipeline(), resolution);
        }
        if (!_prev_frame_view_matrix.valid()) {
            _prev_frame_view_matrix = pipeline().device().create_buffer<float4x4>(1u);
        }
        using namespace luisa::compute;
        Kernel2D generate_sample_kernel = [&](UInt frame_index, Float time, UInt num_initial_sample) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _generate_sample(num_initial_sample, camera, frame_index, pixel_id, time);
        };
        Kernel2D visibility_reuse_kernel = [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _visibility_reuse(frame_index, pixel_id, time);
        };
        Kernel2D temporal_reuse_kernel = [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _temporal_reuse(camera, frame_index, pixel_id, time);
        };
        Kernel2D perturb_sample_kernel = [&](UInt frame_index, Float time) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            _perturb_sample(frame_index, pixel_id, time);
        };
        Kernel2D spatial_reuse_kernel = [&](UInt frame_index, Float time, UInt pass_index) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            $if(pass_index % 2u == 0u) {
                _spatial_reuse(pass_index, camera, frame_index, pixel_id, time);
            }
            $else {
                _spatial_reuse(pass_index, camera, frame_index, pixel_id, time);
            };
        };
        Kernel2D swap_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto r1 = _spatial_reservoir_buffer->read(pixel_id);
            auto r2 = _temporal_reservoir_buffer->read(pixel_id);
            _spatial_reservoir_buffer->write(r2, pixel_id);
            _temporal_reservoir_buffer->write(r1, pixel_id);
        };
        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
            $if(all(pixel_id == 0u)) {
                auto view_matrix = inverse(camera->camera_to_world());
                _prev_frame_view_matrix->write(0u, view_matrix);
            };
        };

        Clock clock_compile;
        auto generate = pipeline().device().compile(generate_sample_kernel);
        auto visibility_reuse = pipeline().device().compile(visibility_reuse_kernel);
        auto temporal_reuse = pipeline().device().compile(temporal_reuse_kernel);
        auto perturb = pipeline().device().compile(perturb_sample_kernel);
        auto spatial_reuse = pipeline().device().compile(spatial_reuse_kernel);
        auto swap = pipeline().device().compile(swap_kernel);
        auto render = pipeline().device().compile(render_kernel);
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        luisa::vector<float4> local_pixels;
        if (node()->video()) {
            shutter_samples = camera->node()->uniform_shutter_samples();
            for(auto &s: shutter_samples) s.spp = 1u;
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
                auto constexpr num_spatial_reuse_pass = 2u;
                camera->film()->clear(command_buffer);
                command_buffer << generate(_total_frame_count, s.point.time, node<ReSTIRDirectLighting>()->num_initial_sample())
                                      .dispatch(resolution);
                if (node<ReSTIRDirectLighting>()->enable_visibility_reuse()) {
                    command_buffer << visibility_reuse(_total_frame_count, s.point.time).dispatch(resolution);
                }
                if (node<ReSTIRDirectLighting>()->enable_temporal_reuse()) {
                    command_buffer << temporal_reuse(_total_frame_count, s.point.time).dispatch(resolution);
                    if (node<ReSTIRDirectLighting>()->enable_decorrelation()) {
                        command_buffer << perturb(_total_frame_count, s.point.time).dispatch(resolution);
                    }
                }
                if (node<ReSTIRDirectLighting>()->enable_spatial_reuse()) {
                    for (auto j = 0u; j < num_spatial_reuse_pass; j++) {
                        command_buffer << spatial_reuse(_total_frame_count, s.point.time, j).dispatch(resolution);
                    }
                }
                command_buffer << swap().dispatch(resolution);
                command_buffer << render(_total_frame_count, s.point.time, s.point.weight)
                                      .dispatch(resolution);
                sample_id++;
                _total_frame_count++;
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
                if (node()->save()) {
                    camera->film()->download(command_buffer, local_pixels.data());
                    command_buffer << synchronize();
                    auto film_path = camera->node()->file();
                    // film_path is a std::filesystem::path, add number to its name
                    auto new_name = film_path.stem().string() + std::format("{:05}", shutter_id) + film_path.extension().string();
                    auto new_film_path = film_path.replace_filename(new_name);
                    save_image(new_film_path, reinterpret_cast<const float *>(local_pixels.data()), resolution);
                }
                camera->film()->clear(command_buffer);
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
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};
        auto ray = _visibility_buffer->ray(pixel_id);
        auto hit = _visibility_buffer->hit(pixel_id);
        auto weight = _visibility_buffer->weight(pixel_id);
        $loop {
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->interaction(ray, hit);
            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += weight * eval.L;
                }
                $break;
            };
            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += weight * eval.L;
                };
            }
            // compute direct lighting
            $if(!it->shape().has_surface()) { $break; };
            auto reservoir = _temporal_reservoir_buffer->read(pixel_id);
            auto [L, _] = _evaluate_with_occlusion(reservoir.sample, *it, wo, swl, time);
            auto contribution_weight = reservoir.contribution_weight();
            Li += weight * contribution_weight * L;
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