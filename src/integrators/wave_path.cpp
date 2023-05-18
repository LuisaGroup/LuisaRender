//
// Created by Mike Smith on 2022/1/10.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/thread_pool.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/display.h>

namespace luisa::render {

using namespace compute;

template<uint dim, typename F>
[[nodiscard]] auto compile_async(Device &device, F &&f) noexcept {
    auto kernel = [&] {
        if constexpr (dim == 1u) {
            return Kernel1D{f};
        } else if constexpr (dim == 2u) {
            return Kernel2D{f};
        } else if constexpr (dim == 3u) {
            return Kernel3D{f};
        } else {
            static_assert(always_false_v<F>, "Invalid dimension.");
        }
    }();
    return global_thread_pool().async([&device, kernel] { return device.compile(kernel); });
}

class WavefrontPathTracing final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _samples_per_pass;

public:
    WavefrontPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _samples_per_pass{std::max(desc->property_uint_or_default("samples_per_pass", 16u), 1u)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto samples_per_pass() const noexcept { return _samples_per_pass; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PathStateSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _wl_sample;
    Buffer<float> _beta;
    Buffer<float> _radiance;
    Buffer<float> _pdf_bsdf;

public:
    PathStateSOA(const Spectrum::Instance *spectrum, size_t size) noexcept
        : _spectrum{spectrum} {
        auto &&device = spectrum->pipeline().device();
        auto dimension = spectrum->node()->dimension();
        _beta = device.create_buffer<float>(size * dimension);
        _radiance = device.create_buffer<float>(size * dimension);
        _pdf_bsdf = device.create_buffer<float>(size);
        if (!spectrum->node()->is_fixed()) {
            _wl_sample = device.create_buffer<float>(size);
        }
    }
    [[nodiscard]] auto read_beta(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _beta->read(offset + i);
        }
        return s;
    }
    void write_beta(Expr<uint> index, const SampledSpectrum &beta) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _beta->write(offset + i, beta[i]);
        }
    }
    [[nodiscard]] auto read_swl(Expr<uint> index) const noexcept {
        if (_spectrum->node()->is_fixed()) {
            return std::make_pair(def(0.f), _spectrum->sample(0.f));
        }
        auto u_wl = _wl_sample->read(index);
        auto swl = _spectrum->sample(abs(u_wl));
        $if(u_wl < 0.f) { swl.terminate_secondary(); };
        return std::make_pair(abs(u_wl), swl);
    }
    void write_wavelength_sample(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample->write(index, u_wl);
        }
    }
    void terminate_secondary_wavelengths(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample->write(index, -u_wl);
        }
    }
    [[nodiscard]] auto read_radiance(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _radiance->read(offset + i);
        }
        return s;
    }
    void write_radiance(Expr<uint> index, const SampledSpectrum &s) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _radiance->write(offset + i, s[i]);
        }
    }
    [[nodiscard]] auto read_pdf_bsdf(Expr<uint> index) const noexcept {
        return _pdf_bsdf->read(index);
    }
    void write_pdf_bsdf(Expr<uint> index, Expr<float> pdf) noexcept {
        _pdf_bsdf->write(index, pdf);
    }
};

class LightSampleSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _emission;
    Buffer<float4> _wi_and_pdf;

public:
    LightSampleSOA(const Spectrum::Instance *spec, size_t size) noexcept
        : _spectrum{spec} {
        auto &&device = spec->pipeline().device();
        auto dimension = spec->node()->dimension();
        _emission = device.create_buffer<float>(size * dimension);
        _wi_and_pdf = device.create_buffer<float4>(size);
    }
    [[nodiscard]] auto read_emission(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _emission->read(offset + i);
        }
        return s;
    }
    void write_emission(Expr<uint> index, const SampledSpectrum &s) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _emission->write(offset + i, s[i]);
        }
    }
    [[nodiscard]] auto read_wi_and_pdf(Expr<uint> index) const noexcept {
        return _wi_and_pdf->read(index);
    }
    void write_wi_and_pdf(Expr<uint> index, Expr<float3> wi, Expr<float> pdf) noexcept {
        _wi_and_pdf->write(index, make_float4(wi, pdf));
    }
};

class RayQueue {

public:
    static constexpr auto counter_buffer_size = 16u * 1024u;

private:
    Buffer<uint> _index_buffer;
    Buffer<uint> _counter_buffer;
    uint _current_counter;
    Shader1D<> _clear_counters;

public:
    RayQueue(Device &device, size_t size) noexcept
        : _index_buffer{device.create_buffer<uint>(size)},
          _counter_buffer{device.create_buffer<uint>(counter_buffer_size)},
          _current_counter{counter_buffer_size} {
        _clear_counters = device.compile<1>([this] {
            _counter_buffer->write(dispatch_x(), 0u);
        });
    }
    [[nodiscard]] BufferView<uint> prepare_counter_buffer(CommandBuffer &command_buffer) noexcept {
        if (_current_counter == counter_buffer_size) {
            _current_counter = 0u;
            command_buffer << _clear_counters().dispatch(counter_buffer_size);
        }
        return _counter_buffer.view(_current_counter++, 1u);
    }
    [[nodiscard]] BufferView<uint> prepare_index_buffer(CommandBuffer &command_buffer) noexcept {
        return _index_buffer;
    }
};

class WavefrontPathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override;
};

luisa::unique_ptr<Integrator::Instance> WavefrontPathTracing::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<WavefrontPathTracingInstance>(pipeline, command_buffer, this);
}

void WavefrontPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {

    auto &&device = camera->pipeline().device();
    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    // determine configurations
    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto pixel_count = resolution.x * resolution.y;
    auto max_samples_per_pass = ((1ull << 30u) + pixel_count - 1u) / pixel_count;
    auto samples_per_pass = std::min(node<WavefrontPathTracing>()->samples_per_pass(),
                                     static_cast<uint32_t>(max_samples_per_pass));
    auto state_count = static_cast<uint64_t>(samples_per_pass) *
                       static_cast<uint64_t>(pixel_count);
    LUISA_INFO("Wavefront path tracing configurations: "
               "resolution = {}x{}, spp = {}, state_count = {}, samples_per_pass = {}.",
               resolution.x, resolution.y, spp, state_count, samples_per_pass);

    auto spectrum = pipeline().spectrum();
    PathStateSOA path_states{spectrum, state_count};
    LightSampleSOA light_samples{spectrum, state_count};
    sampler()->reset(command_buffer, resolution, state_count, spp);
    command_buffer << synchronize();

    using BufferRay = BufferVar<Ray>;
    using BufferHit = BufferVar<Hit>;

    LUISA_INFO("Compiling ray generation kernel.");
    Clock clock_compile;
    auto generate_rays_shader = compile_async<1>(device, [&](BufferUInt path_indices, BufferRay rays,
                                                             UInt base_sample_id, Float time) noexcept {
        auto state_id = dispatch_x();
        auto pixel_id = state_id % pixel_count;
        auto sample_id = base_sample_id + state_id / pixel_count;
        auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
        sampler()->start(pixel_coord, sample_id);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto u_wavelength = spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d();
        sampler()->save_state(state_id);
        auto camera_sample = camera->generate_ray(pixel_coord, time, u_filter, u_lens);
        rays.write(state_id, camera_sample.ray);
        path_states.write_wavelength_sample(state_id, u_wavelength);
        path_states.write_beta(state_id, SampledSpectrum{spectrum->node()->dimension(), camera_sample.weight});
        path_states.write_radiance(state_id, SampledSpectrum{spectrum->node()->dimension()});
        path_states.write_pdf_bsdf(state_id, 1e16f);
        path_indices.write(state_id, state_id);
    });

    LUISA_INFO("Compiling intersection kernel.");
    auto intersect_shader = compile_async<1>(device, [&](BufferUInt ray_count, BufferRay rays, BufferHit hits,
                                                         BufferUInt surface_queue, BufferUInt surface_queue_size,
                                                         BufferUInt light_queue, BufferUInt light_queue_size,
                                                         BufferUInt escape_queue, BufferUInt escape_queue_size) noexcept {
        auto ray_id = dispatch_x();
        $if(ray_id < ray_count.read(0u)) {
            auto ray = rays.read(ray_id);
            auto hit = pipeline().geometry()->trace_closest(ray);
            hits.write(ray_id, hit);
            $if(!hit->miss()) {
                auto shape = pipeline().geometry()->instance(hit.inst);
                $if(shape.has_surface()) {
                    auto queue_id = surface_queue_size.atomic(0u).fetch_add(1u);
                    surface_queue.write(queue_id, ray_id);
                };
                $if(shape.has_light()) {
                    auto queue_id = light_queue_size.atomic(0u).fetch_add(1u);
                    light_queue.write(queue_id, ray_id);
                };
            }
            $else {
                if (pipeline().environment()) {
                    auto queue_id = escape_queue_size.atomic(0u).fetch_add(1u);
                    escape_queue.write(queue_id, ray_id);
                }
            };
        };
    });

    LUISA_INFO("Compiling environment evaluation kernel.");
    auto evaluate_miss_shader = compile_async<1>(device, [&](BufferUInt path_indices, BufferRay rays,
                                                             BufferUInt queue, BufferUInt queue_size, Float time) noexcept {
        if (pipeline().environment()) {
            auto queue_id = dispatch_x();
            $if(queue_id < queue_size.read(0u)) {
                auto ray_id = queue.read(queue_id);
                auto wi = rays.read(ray_id)->direction();
                auto path_id = path_indices.read(ray_id);
                auto [u_wl, swl] = path_states.read_swl(path_id);
                auto pdf_bsdf = path_states.read_pdf_bsdf(path_id);
                auto beta = path_states.read_beta(path_id);
                auto Li = path_states.read_radiance(path_id);
                auto eval = light_sampler()->evaluate_miss(wi, swl, time);
                auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                Li += beta * eval.L * mis_weight;
                path_states.write_radiance(path_id, Li);
            };
        }
    });

    LUISA_INFO("Compiling light evaluation kernel.");
    auto evaluate_light_shader = compile_async<1>(device, [&](BufferUInt path_indices, BufferRay rays, BufferHit hits,
                                                              BufferUInt queue, BufferUInt queue_size, Float time) noexcept {
        if (!pipeline().lights().empty()) {
            auto queue_id = dispatch_x();
            $if(queue_id < queue_size.read(0u)) {
                auto ray_id = queue.read(queue_id);
                auto ray = rays.read(ray_id);
                auto hit = hits.read(ray_id);
                auto path_id = path_indices.read(ray_id);
                auto [u_wl, swl] = path_states.read_swl(path_id);
                auto pdf_bsdf = path_states.read_pdf_bsdf(path_id);
                auto beta = path_states.read_beta(path_id);
                auto Li = path_states.read_radiance(path_id);
                auto it = pipeline().geometry()->interaction(ray, hit);
                auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                Li += beta * eval.L * mis_weight;
                path_states.write_radiance(path_id, Li);
            };
        }
    });

    LUISA_INFO("Compiling light sampling kernel.");
    auto sample_light_shader = compile_async<1>(device, [&](BufferUInt path_indices, BufferRay rays, BufferHit hits,
                                                            BufferUInt queue, BufferUInt queue_size, Float time) noexcept {
        auto queue_id = dispatch_x();
        $if(queue_id < queue_size.read(0u)) {
            auto ray_id = queue.read(queue_id);
            auto path_id = path_indices.read(ray_id);
            sampler()->load_state(path_id);
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            sampler()->save_state(path_id);
            auto ray = rays.read(ray_id);
            auto hit = hits.read(ray_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto [u_wl, swl] = path_states.read_swl(path_id);
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);
            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
            light_samples.write_emission(queue_id, ite(occluded, 0.f, 1.f) * light_sample.eval.L);
            light_samples.write_wi_and_pdf(queue_id, light_sample.shadow_ray->direction(),
                                           ite(occluded, 0.f, light_sample.eval.pdf));
        };
    });

    LUISA_INFO("Compiling surface evaluation kernel.");
    auto evaluate_surface_shader = compile_async<1>(device, [&](BufferUInt path_indices, UInt trace_depth, BufferUInt queue, BufferUInt queue_size,
                                                                BufferRay in_rays, BufferHit in_hits, BufferRay out_rays,
                                                                BufferUInt out_queue, BufferUInt out_queue_size, Float time) noexcept {
        auto queue_id = dispatch_x();
        $if(queue_id < queue_size.read(0u)) {
            auto ray_id = queue.read(queue_id);
            auto path_id = path_indices.read(ray_id);
            sampler()->load_state(path_id);
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<WavefrontPathTracing>()->rr_depth();
            $if(trace_depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            sampler()->save_state(path_id);
            auto ray = in_rays.read(ray_id);
            auto hit = in_hits.read(ray_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto u_wl_and_swl = path_states.read_swl(path_id);
            auto &&u_wl = u_wl_and_swl.first;
            auto &&swl = u_wl_and_swl.second;
            auto beta = path_states.read_beta(path_id);
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            auto wo = -ray->direction();

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
                    path_states.write_pdf_bsdf(path_id, 1e16f);
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) {
                            swl.terminate_secondary();
                            path_states.terminate_secondary_wavelengths(path_id, u_wl);
                        };
                    }
                    // direct lighting
                    auto light_wi_and_pdf = light_samples.read_wi_and_pdf(queue_id);
                    auto pdf_light = light_wi_and_pdf.w;
                    $if(light_wi_and_pdf.w > 0.f) {
                        auto eval = closure->evaluate(wo, light_wi_and_pdf.xyz());
                        auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                        // update Li
                        auto Ld = light_samples.read_emission(queue_id);
                        auto Li = path_states.read_radiance(path_id);
                        Li += mis_weight / pdf_light * beta * eval.f * Ld;
                        path_states.write_radiance(path_id, Li);
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    path_states.write_pdf_bsdf(path_id, surface_sample.eval.pdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    auto w = ite(surface_sample.eval.pdf > 0.0f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
                    };
                };
            });

            // prepare for next bounce
            auto terminated = def(false);
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) {
                terminated = true;
            }
            $else {
                // rr
                auto rr_threshold = node<WavefrontPathTracing>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, 0.05f);
                $if(trace_depth + 1u >= rr_depth) {
                    terminated = q < rr_threshold & u_rr >= q;
                    beta *= ite(q < rr_threshold, 1.f / q, 1.f);
                };
            };
            $if(!terminated) {
                auto out_queue_id = out_queue_size.atomic(0u).fetch_add(1u);
                out_queue.write(out_queue_id, path_id);
                out_rays.write(out_queue_id, ray);
                path_states.write_beta(path_id, beta);
            };
        };
    });

    LUISA_INFO("Compiling accumulation kernel.");
    auto accumulate_shader = compile_async<1>(device, [&](Float shutter_weight) noexcept {
        auto state_id = dispatch_x();
        auto pixel_id = state_id % pixel_count;
        auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
        auto [u_wl, swl] = path_states.read_swl(state_id);
        auto Li = path_states.read_radiance(state_id);
        camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li * shutter_weight));
    });

    // wait for the compilation of all shaders
    generate_rays_shader.wait();
    intersect_shader.wait();
    evaluate_miss_shader.wait();
    evaluate_surface_shader.wait();
    evaluate_light_shader.wait();
    sample_light_shader.wait();
    accumulate_shader.wait();
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);

    LUISA_INFO("Rendering started.");
    // create path states
    RayQueue path_queue{device, state_count};
    RayQueue out_path_queue{device, state_count};
    RayQueue surface_queue{device, state_count};
    RayQueue light_queue{device, state_count};
    RayQueue miss_queue{device, state_count};
    auto ray_buffer = device.create_buffer<Ray>(state_count);
    auto ray_buffer_out = device.create_buffer<Ray>(state_count);
    auto hit_buffer = device.create_buffer<Hit>(state_count);
    auto state_count_buffer = device.create_buffer<uint>(samples_per_pass);
    luisa::vector<uint> precomputed_state_counts(samples_per_pass);
    for (auto i = 0u; i < samples_per_pass; i++) {
        precomputed_state_counts[i] = (i + 1u) * pixel_count;
    }
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << state_count_buffer.copy_from(precomputed_state_counts.data())
                   << synchronize();

    auto sample_id = 0u;
    auto last_committed_sample_id = 0u;
    Clock clock;
    ProgressBar progress_bar;
    progress_bar.update(0.0);
    for (auto s : shutter_samples) {
        auto time = s.point.time;
        pipeline().update(command_buffer, time);
        for (auto i = 0u; i < s.spp; i += samples_per_pass) {
            auto launch_spp = std::min(s.spp - i, samples_per_pass);
            auto launch_state_count = launch_spp * pixel_count;
            auto path_indices = path_queue.prepare_index_buffer(command_buffer);
            auto path_count = state_count_buffer.view(launch_spp - 1u, 1u);
            auto rays = ray_buffer.view();
            auto hits = hit_buffer.view();
            auto out_rays = ray_buffer_out.view();
            command_buffer << generate_rays_shader.get()(path_indices, rays, sample_id, time)
                                  .dispatch(launch_state_count);
            for (auto depth = 0u; depth < node<WavefrontPathTracing>()->max_depth(); depth++) {
                auto surface_indices = surface_queue.prepare_index_buffer(command_buffer);
                auto surface_count = surface_queue.prepare_counter_buffer(command_buffer);
                auto light_indices = light_queue.prepare_index_buffer(command_buffer);
                auto light_count = light_queue.prepare_counter_buffer(command_buffer);
                auto miss_indices = miss_queue.prepare_index_buffer(command_buffer);
                auto miss_count = miss_queue.prepare_counter_buffer(command_buffer);
                auto out_path_indices = out_path_queue.prepare_index_buffer(command_buffer);
                auto out_path_count = out_path_queue.prepare_counter_buffer(command_buffer);
                command_buffer << intersect_shader.get()(path_count, rays, hits, surface_indices, surface_count,
                                                         light_indices, light_count, miss_indices, miss_count)
                                      .dispatch(launch_state_count);
                if (pipeline().environment()) {
                    command_buffer << evaluate_miss_shader.get()(path_indices, rays, miss_indices, miss_count, time)
                                          .dispatch(launch_state_count);
                }
                if (!pipeline().lights().empty()) {
                    command_buffer << evaluate_light_shader.get()(path_indices, rays, hits,
                                                                  light_indices, light_count, time)
                                          .dispatch(launch_state_count);
                }
                command_buffer << sample_light_shader.get()(path_indices, rays, hits, surface_indices, surface_count, time)
                                      .dispatch(launch_state_count)
                               << evaluate_surface_shader.get()(path_indices, depth, surface_indices,
                                                                surface_count, rays, hits, out_rays,
                                                                out_path_indices, out_path_count, time)
                                      .dispatch(launch_state_count);
                path_indices = out_path_indices;
                path_count = out_path_count;
                std::swap(rays, out_rays);
                std::swap(path_queue, out_path_queue);
            }
            command_buffer << accumulate_shader.get()(s.point.weight).dispatch(launch_state_count);
            sample_id += launch_spp;
            auto launches_per_commit =
                display() && !display()->should_close() ?
                    node<WavefrontPathTracing>()->display_interval() :
                    16u;
            if (sample_id - last_committed_sample_id >= launches_per_commit) {
                last_committed_sample_id = sample_id;
                auto p = sample_id / static_cast<double>(spp);
                if (display() && display()->update(command_buffer, sample_id)) {
                    progress_bar.update(p);
                } else {
                    command_buffer << [p, &progress_bar] { progress_bar.update(p); };
                }
            }
        }
    }
    command_buffer << synchronize();
    progress_bar.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WavefrontPathTracing)
