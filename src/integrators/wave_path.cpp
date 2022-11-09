//
// Created by Mike Smith on 2022/1/10.
//

#include <fstream>

#include <luisa-compute.h>
#include <util/imageio.h>
#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class WavefrontPathTracing final : public Integrator {

public:
    static constexpr auto min_state_count = 1024u * 1024u;
    static constexpr auto default_state_count = 16u * min_state_count;

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _state_count;

public:
    WavefrontPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _state_count{std::max(desc->property_uint_or_default("states", default_state_count), min_state_count)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto state_count() const noexcept { return _state_count; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PathStateSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _swl_lambda;
    Buffer<float> _swl_pdf;
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
            _swl_lambda = device.create_buffer<float>(size * dimension);
            _swl_pdf = device.create_buffer<float>(size * dimension);
        }
    }
    [[nodiscard]] auto read_beta(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _beta.read(offset + i);
        }
        return s;
    }
    void write_beta(Expr<uint> index, const SampledSpectrum &beta) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _beta.write(offset + i, beta[i]);
        }
    }
    [[nodiscard]] auto read_swl(Expr<uint> index) const noexcept {
        if (_spectrum->node()->is_fixed()) { return _spectrum->sample(0.f); }
        SampledWavelengths swl{_spectrum->node()->dimension()};
        auto offset = index * swl.dimension();
        for (auto i = 0u; i < swl.dimension(); i++) {
            swl.set_lambda(i, _swl_lambda.read(offset + i));
            swl.set_pdf(i, _swl_pdf.read(offset + i));
        }
        return swl;
    }
    void write_swl(Expr<uint> index, const SampledWavelengths &swl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            auto offset = index * swl.dimension();
            for (auto i = 0u; i < swl.dimension(); i++) {
                _swl_lambda.write(offset + i, swl.lambda(i));
                _swl_pdf.write(offset + i, swl.pdf(i));
            }
        }
    }
    [[nodiscard]] auto read_radiance(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _radiance.read(offset + i);
        }
        return s;
    }
    void write_radiance(Expr<uint> index, const SampledSpectrum &s) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _radiance.write(offset + i, s[i]);
        }
    }
    [[nodiscard]] auto read_pdf_bsdf(Expr<uint> index) const noexcept {
        return _pdf_bsdf.read(index);
    }
    void write_pdf_bsdf(Expr<uint> index, Expr<float> pdf) noexcept {
        _pdf_bsdf.write(index, pdf);
    }
};

class LightSampleSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _emission;
    Buffer<float> _pdf;
    Buffer<float3> _wi;

public:
    LightSampleSOA(const Spectrum::Instance *spec, size_t size) noexcept
        : _spectrum{spec} {
        auto &&device = spec->pipeline().device();
        auto dimension = spec->node()->dimension();
        _emission = device.create_buffer<float>(size * dimension);
        _pdf = device.create_buffer<float>(size);
        _wi = device.create_buffer<float3>(size);
    }
    [[nodiscard]] auto read_emission(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _emission.read(offset + i);
        }
        return s;
    }
    void write_emission(Expr<uint> index, const SampledSpectrum &s) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _emission.write(offset + i, s[i]);
        }
    }
    [[nodiscard]] auto read_pdf(Expr<uint> index) const noexcept {
        return _pdf.read(index);
    }
    void write_pdf(Expr<uint> index, Expr<float> pdf) noexcept {
        _pdf.write(index, pdf);
    }
    [[nodiscard]] auto read_wi(Expr<uint> index) const noexcept {
        return _wi.read(index);
    }
    void write_wi(Expr<uint> index, Expr<float3> wi) const noexcept {
        _wi.write(index, wi);
    }
};

class RayQueue {

public:
    static constexpr auto counter_buffer_size = 1024u;

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
            _counter_buffer.write(dispatch_x(), 0u);
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

class WavefrontPathTracingInstance final : public Integrator::Instance {

private:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

public:
    WavefrontPathTracingInstance(const WavefrontPathTracing *node, Pipeline &pipeline, CommandBuffer &cb) noexcept
        : Integrator::Instance{pipeline, cb, node} {}

    void render(Stream &stream) noexcept override {
        auto pt = node<WavefrontPathTracing>();
        auto command_buffer = stream.command_buffer();
        luisa::vector<float4> pixels;
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            pixels.resize(next_pow2(pixel_count));
            auto film_path = camera->node()->file();
            LUISA_INFO(
                "Rendering to '{}' of resolution {}x{} at {}spp.",
                film_path.string(),
                resolution.x, resolution.y,
                camera->node()->spp());
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, camera);
            camera->film()->download(command_buffer, pixels.data());
            command_buffer << compute::synchronize();
            camera->film()->release();
            save_image(film_path, (const float *)pixels.data(), resolution);
        }
    }
};

luisa::unique_ptr<Integrator::Instance> WavefrontPathTracing::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<WavefrontPathTracingInstance>(this, pipeline, command_buffer);
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
    auto state_count = node<WavefrontPathTracing>()->state_count();
    auto spp_per_launch = (state_count + pixel_count - 1u) / pixel_count;
    state_count = spp_per_launch * pixel_count;
    LUISA_INFO("Wavefront path tracing configurations: "
               "resolution = {}x{}, spp = {}, state_count = {}, spp_per_launch = {}.",
               resolution.x, resolution.y, spp, state_count, spp_per_launch);

    auto spectrum = pipeline().spectrum();
    PathStateSOA path_states{spectrum, state_count};
    LightSampleSOA light_samples{spectrum, state_count};
    sampler()->reset(command_buffer, resolution, state_count, spp);
    command_buffer << synchronize();

    using BufferRay = BufferVar<Ray>;
    using BufferHit = BufferVar<Hit>;

    LUISA_INFO("Compiling ray generation kernel.");
    Clock clock_compile;
    auto generate_rays_shader = device.compile_async<1>([&](BufferUInt path_indices, BufferRay rays,
                                                            UInt base_sample_id, Float time) noexcept {
        auto state_id = dispatch_x();
        auto pixel_id = state_id % pixel_count;
        auto sample_id = base_sample_id + state_id / pixel_count;
        auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
        sampler()->start(pixel_coord, sample_id);
        auto camera_sample = camera->generate_ray(*sampler(), pixel_coord, time);
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        sampler()->save_state(state_id);
        rays.write(state_id, camera_sample.ray);
        path_states.write_swl(state_id, swl);
        path_states.write_beta(state_id, SampledSpectrum{swl.dimension(), camera_sample.weight});
        path_states.write_radiance(state_id, SampledSpectrum{swl.dimension()});
        path_states.write_pdf_bsdf(state_id, 1e16f);
        path_indices.write(state_id, state_id);
    });

    LUISA_INFO("Compiling intersection kernel.");
    auto intersect_shader = device.compile_async<1>([&](BufferUInt ray_count, BufferRay rays, BufferHit hits,
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
                $if(shape->has_surface()) {
                    auto queue_id = surface_queue_size.atomic(0u).fetch_add(1u);
                    surface_queue.write(queue_id, ray_id);
                };
                $if(shape->has_light()) {
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
    auto evaluate_miss_shader = device.compile_async<1>([&](BufferUInt path_indices, BufferRay rays,
                                                            BufferUInt queue, BufferUInt queue_size, Float time) noexcept {
        if (pipeline().environment()) {
            auto queue_id = dispatch_x();
            $if(queue_id < queue_size.read(0u)) {
                auto ray_id = queue.read(queue_id);
                auto wi = rays.read(ray_id)->direction();
                auto path_id = path_indices.read(ray_id);
                auto swl = path_states.read_swl(path_id);
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
    auto evaluate_light_shader = device.compile_async<1>([&](BufferUInt path_indices, BufferRay rays, BufferHit hits,
                                                             BufferUInt queue, BufferUInt queue_size, Float time) noexcept {
        if (!pipeline().lights().empty()) {
            auto queue_id = dispatch_x();
            $if(queue_id < queue_size.read(0u)) {
                auto ray_id = queue.read(queue_id);
                auto ray = rays.read(ray_id);
                auto hit = hits.read(ray_id);
                auto path_id = path_indices.read(ray_id);
                auto swl = path_states.read_swl(path_id);
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
    auto sample_light_shader = device.compile_async<1>([&](BufferUInt path_indices, BufferRay rays, BufferHit hits,
                                                           BufferUInt queue, BufferUInt queue_size, Float time) noexcept {
        auto queue_id = dispatch_x();
        $if(queue_id < queue_size.read(0u)) {
            auto ray_id = queue.read(queue_id);
            auto ray = rays.read(ray_id);
            auto hit = hits.read(ray_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto path_id = path_indices.read(ray_id);
            auto swl = path_states.read_swl(path_id);
            sampler()->load_state(path_id);
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            Light::Sample light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);
            sampler()->save_state(path_id);
            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.ray);
            light_samples.write_emission(queue_id, ite(occluded, 0.f, 1.f) * light_sample.eval.L);
            light_samples.write_pdf(queue_id, ite(occluded, 0.f, light_sample.eval.pdf));
            light_samples.write_wi(queue_id, light_sample.ray->direction());
        };
    });

    LUISA_INFO("Compiling surface evaluation kernel.");
    auto evaluate_surface_shader = device.compile_async<1>([&](BufferUInt path_indices, UInt trace_depth, BufferUInt queue, BufferUInt queue_size,
                                                               BufferRay in_rays, BufferHit in_hits, BufferRay out_rays,
                                                               BufferUInt out_queue, BufferUInt out_queue_size, Float time) noexcept {
        auto queue_id = dispatch_x();
        $if(queue_id < queue_size.read(0u)) {
            auto ray_id = queue.read(queue_id);
            auto ray = in_rays.read(ray_id);
            auto hit = in_hits.read(ray_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto path_id = path_indices.read(ray_id);
            sampler()->load_state(path_id);
            auto Li = path_states.read_radiance(path_id);
            auto swl = path_states.read_swl(path_id);
            auto beta = path_states.read_beta(path_id);
            auto surface_tag = it->shape()->surface_tag();
            auto pdf_bsdf = def(0.f);
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto eta = def(1.f);
            auto eta_scale = def(1.f);
            auto wo = -ray->direction();
            auto surface_sample = Surface::Sample::zero(swl.dimension());
            auto alpha_skip = def(false);
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);

                // apply roughness map
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(!alpha_skip) {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) {
                            swl.terminate_secondary();
                            path_states.write_swl(path_id, swl);
                        };
                    }
                    // direct lighting
                    auto pdf_light = light_samples.read_pdf(queue_id);
                    $if(pdf_light > 0.0f) {
                        auto Ld = light_samples.read_emission(queue_id);
                        auto wi = light_samples.read_wi(queue_id);
                        auto eval = closure->evaluate(wo, wi);
                        auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                        Li += mis_weight / pdf_light * beta * eval.f * Ld;
                    };
                    // sample material
                    surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    eta = closure->eta().value_or(1.f);
                };
            });
            path_states.write_radiance(path_id, Li);

            // prepare for the next bounce
            auto terminated = def(false);
            $if(alpha_skip) {
                ray = it->spawn_ray(ray->direction());
                pdf_bsdf = 1e16f;
            }
            $else {
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.0f, 1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
                };
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) {
                    terminated = true;
                }
                $else {
                    auto rr_depth = node<WavefrontPathTracing>()->rr_depth();
                    auto rr_threshold = node<WavefrontPathTracing>()->rr_threshold();
                    // rr
                    auto q = max(beta.max() * eta_scale, 0.05f);
                    $if(trace_depth + 1u >= rr_depth & q < rr_threshold) {
                        auto u = sampler()->generate_1d();
                        terminated = u >= q;
                        beta *= 1.f / q;
                    };
                };
            };
            $if(!terminated) {
                auto out_queue_id = out_queue_size.atomic(0u).fetch_add(1u);
                out_queue.write(out_queue_id, path_id);
                out_rays.write(out_queue_id, ray);
                sampler()->save_state(path_id);
                path_states.write_beta(path_id, beta);
                path_states.write_pdf_bsdf(path_id, pdf_bsdf);
            };
        };
    });

    LUISA_INFO("Compiling accumulation kernel.");
    auto accumulate_shader = device.compile_async<1>([&](Float shutter_weight) noexcept {
        auto state_id = dispatch_x();
        auto pixel_id = state_id % pixel_count;
        auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
        auto swl = path_states.read_swl(state_id);
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
    auto state_count_buffer = device.create_buffer<uint>(spp_per_launch);
    luisa::vector<uint> precomputed_state_counts(spp_per_launch);
    for (auto i = 0; i < spp_per_launch; i++) {
        precomputed_state_counts[i] = (i + 1u) * pixel_count;
    }
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << state_count_buffer.copy_from(precomputed_state_counts.data())
                   << synchronize();

    auto sample_id = 0u;
    auto last_committed_sample_id = 0u;
    constexpr auto launches_per_commit = 16u;
    Clock clock;
    ProgressBar progress_bar;
    progress_bar.update(0.0);
    for (auto s : shutter_samples) {
        auto time = s.point.time;
        pipeline().update(command_buffer, time);
        for (auto i = 0u; i < s.spp; i += spp_per_launch) {
            auto launch_spp = std::min(s.spp - i, spp_per_launch);
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
            if (sample_id - last_committed_sample_id >= launches_per_commit) {
                last_committed_sample_id = sample_id;
                auto p = sample_id / static_cast<double>(spp);
                command_buffer << [p, &progress_bar] { progress_bar.update(p); };
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
