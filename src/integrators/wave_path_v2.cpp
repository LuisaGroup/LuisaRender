//
// Created by Mike Smith on 2022/1/10.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/display.h>

namespace luisa::render {

using namespace compute;
enum KernelState {
    INVALID = 0,
    INTERSECT,
    MISS,
    LIGHT,
    SAMPLE,
    SURFACE,
    KERNEL_COUNT
};
const luisa::string KernelName[KERNEL_COUNT] = {"INVALID",
                             "INTERSECT",
                             "MISS",
                             "LIGHT",
                             "SAMPLE",
                             "SURFACE"};
class WavefrontPathTracingv2 final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _state_limit;
    bool _gathering;

public:
    WavefrontPathTracingv2(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _state_limit{std::max(desc->property_uint_or_default("state_limit", 16*1024u*1024u), 1024u)},
          _gathering{desc->property_bool_or_default("gathering",false)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto state_limit() const noexcept { return _state_limit; }
    [[nodiscard]] auto gathering() const noexcept { return _gathering; }
	[[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PathStateSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _wl_sample;
    Buffer<float> _beta;
    Buffer<float> _pdf_bsdf;
    //Buffer<int> _kernel_index;
    Buffer<uint> _depth;
    Buffer<uint> _pixel_index;
    //Buffer<uint> _kernel_count;
    luisa::vector<uint> _host_count;
    Buffer<Ray> _ray;
    Buffer<Hit> _hit;
public:
    
    PathStateSOA(const Spectrum::Instance *spectrum, size_t size) noexcept
        : _spectrum{spectrum} {
        auto &&device = spectrum->pipeline().device();
        auto dimension = spectrum->node()->dimension();
        _beta = device.create_buffer<float>(size * dimension);
        _pdf_bsdf = device.create_buffer<float>(size);
        //_kernel_index = device.create_buffer<int>(size);
        //_kernel_count = device.create_buffer<uint>(KERNEL_COUNT);
        //_host_count.resize(KERNEL_COUNT);
        _ray = device.create_buffer<Ray>(size);
        _hit = device.create_buffer<Hit>(size);
        _depth = device.create_buffer<uint>(size);
        _pixel_index = device.create_buffer<uint>(size);
        if (!spectrum->node()->is_fixed()) {
            _wl_sample = device.create_buffer<float>(size);
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
    [[nodiscard]] auto read_ray(Expr<uint> index) const noexcept {
		return _ray.read(index);
	}
    [[nodiscard]] auto read_hit(Expr<uint> index) const noexcept {
		return _hit.read(index);
	}
    void write_ray(Expr<uint> index, Expr<Ray> ray) noexcept {
        _ray.write(index, ray);
    }
    void write_hit(Expr<uint> index, Expr<Hit> hit) noexcept {
        _hit.write(index, hit);
    }
    [[nodiscard]] auto read_depth(Expr<uint> index) const noexcept {
		return _depth.read(index);
	}
    [[nodiscard]] auto read_pixel_index(Expr<uint> index) const noexcept {
        return _pixel_index.read(index);
    }
    void write_pixel_index(Expr<uint> index, Expr<uint> pixel_index) noexcept {
		_pixel_index.write(index, pixel_index);
	}
    void write_depth(Expr<uint> index, Expr<uint> depth) noexcept {
        _depth.write(index, depth);
    }

    void write_beta(Expr<uint> index, const SampledSpectrum &beta) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _beta.write(offset + i, beta[i]);
        }
    }
    [[nodiscard]] auto read_swl(Expr<uint> index) const noexcept {
        if (_spectrum->node()->is_fixed()) {
            return std::make_pair(def(0.f), _spectrum->sample(0.f));
        }
        auto u_wl = _wl_sample.read(index);
        auto swl = _spectrum->sample(abs(u_wl));
        $if (u_wl < 0.f) { swl.terminate_secondary(); };
        return std::make_pair(abs(u_wl), swl);
    }
    void write_wavelength_sample(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample.write(index, u_wl);
        }
    }
    void terminate_secondary_wavelengths(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample.write(index, -u_wl);
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
    [[nodiscard]] auto read_wi_and_pdf(Expr<uint> index) const noexcept {
        return _wi_and_pdf.read(index);
    }
    void write_wi_and_pdf(Expr<uint> index, Expr<float3> wi, Expr<float> pdf) noexcept {
        _wi_and_pdf.write(index, make_float4(wi, pdf));
    }
};

class RayQueue {

public:
    static constexpr auto counter_buffer_size = 1u;

private:
    Buffer<uint> _index_buffer;
    Buffer<uint> _counter_buffer;
    uint _current_counter;
    Shader1D<> _clear_counters;

    uint _host_counter;
public:
    RayQueue(Device &device, size_t size) noexcept
        : _index_buffer{device.create_buffer<uint>(size)},
          _counter_buffer{device.create_buffer<uint>(counter_buffer_size)},
          _current_counter{counter_buffer_size} {
        _clear_counters = device.compile<1>([this] {
            _counter_buffer.write(dispatch_x(), 0u);
        });
    }
    void clear_counter_buffer(CommandBuffer &command_buffer) noexcept {
        //if (_current_counter == counter_buffer_size-1) {
         //   _current_counter = 0u;
            command_buffer << _clear_counters().dispatch(counter_buffer_size);
        //} else
        //    _current_counter++;
    }
    [[nodiscard]] BufferView<uint> counter_buffer(CommandBuffer &command_buffer) noexcept {
        return _counter_buffer;
    }
    [[nodiscard]] BufferView<uint> index_buffer(CommandBuffer &command_buffer) noexcept {
        return _index_buffer;
    }
    [[nodiscard]] uint host_counter() noexcept {
        return _host_counter;
    }
    [[nodiscard]] void catch_counter(CommandBuffer &command_buffer) noexcept {
        
        command_buffer << _counter_buffer.view(0, 1).copy_to(&_host_counter);
    }
};

class WavefrontPathTracingv2Instance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override;
};

luisa::unique_ptr<Integrator::Instance> WavefrontPathTracingv2::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<WavefrontPathTracingv2Instance>(pipeline, command_buffer, this);
}

void WavefrontPathTracingv2Instance::_render_one_camera(
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
    auto state_count = node<WavefrontPathTracingv2>()->state_limit();
    auto gathering = node<WavefrontPathTracingv2>()->gathering();
    LUISA_INFO("Wavefront path tracing configurations: "
               "resolution = {}x{}, spp = {}, state_count = {}.",
               resolution.x, resolution.y, spp, state_count);

    auto spectrum = pipeline().spectrum();
    PathStateSOA path_states{spectrum, state_count};
    LightSampleSOA light_samples{spectrum, state_count};
    sampler()->reset(command_buffer, resolution, state_count, spp);
    command_buffer << synchronize();
    RayQueue queues[KERNEL_COUNT] = {{device, state_count}, {device, state_count}, {device, state_count}, {device, state_count}, {device, state_count}, {device, state_count}};
    LUISA_INFO("Compiling ray generation kernel.");
    Clock clock_compile;
    auto generate_rays_shader = device.compile_async<1>([&](BufferUInt path_indices, BufferUInt intersect_indices, BufferUInt intersect_size,
                                                            UInt base_spp, UInt extra_sample_id, Float time, Float shutter_weight) noexcept {
        auto dispatch_id = dispatch_x();
        auto pixel_id = (extra_sample_id+dispatch_id) % pixel_count;
        auto sample_id = base_spp + (extra_sample_id+dispatch_id) / pixel_count;
        auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
        auto path_id = path_indices.read(dispatch_id);
        sampler()->start(pixel_coord, sample_id);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto u_wavelength = spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d();
        sampler()->save_state(path_id);
        auto camera_sample = camera->generate_ray(pixel_coord, time, u_filter, u_lens);
        
        path_states.write_ray(path_id, camera_sample.ray);
        path_states.write_wavelength_sample(path_id, u_wavelength);
        path_states.write_beta(path_id, SampledSpectrum{spectrum->node()->dimension(), shutter_weight * camera_sample.weight});
        path_states.write_pdf_bsdf(path_id, 1e16f);
        path_states.write_pixel_index(path_id, pixel_id);
        path_states.write_depth(path_id, 0u);
        auto queue_id = intersect_size.atomic(0u).fetch_add(1u);
        if (!gathering)
            intersect_indices.write(queue_id,path_id);
        camera->film()->accumulate(pixel_coord, make_float3(0.f), 1.f);
    });

    LUISA_INFO("Compiling intersection kernel.");
    auto intersect_shader = device.compile_async<1>([&](BufferUInt intersect_indices, BufferUInt surface_queue, BufferUInt surface_queue_size,
                                                        BufferUInt light_queue, BufferUInt light_queue_size,
                                                        BufferUInt escape_queue, BufferUInt escape_queue_size,
                                                        BufferUInt invalid_queue, BufferUInt invalid_queue_size) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = intersect_indices.read(dispatch_id);
        auto ray = path_states.read_ray(path_id);
        auto hit = pipeline().geometry()->trace_closest(ray);
        path_states.write_hit(path_id, hit);
        $if(!hit->miss()) {
            auto shape = pipeline().geometry()->instance(hit.inst);
            
            $if(shape.has_light()) {
                auto queue_id = light_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    light_queue.write(queue_id, path_id);
            }
            $else {
                $if(shape.has_surface()) {
                    auto queue_id = surface_queue_size.atomic(0u).fetch_add(1u);
                    if (!gathering)
                        surface_queue.write(queue_id, path_id);
                }
                $else {
                    auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                    if(!gathering)
                       invalid_queue.write(queue_id, path_id);
                };
            };
        }
        $else {
            if (pipeline().environment()) {
                auto queue_id = escape_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    escape_queue.write(queue_id, path_id);
            } else {
                auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    invalid_queue.write(queue_id, path_id);
            }
        };
    });

    LUISA_INFO("Compiling environment evaluation kernel.");
    auto evaluate_miss_shader = device.compile_async<1>([&](BufferUInt miss_indices, 
                                                            BufferUInt invalid_queue,BufferUInt invalid_queue_size,Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = miss_indices.read(dispatch_id);    
        if (pipeline().environment()) {
            
            auto wi = path_states.read_ray(path_id)->direction();
            auto [u_wl, swl] = path_states.read_swl(path_id);
            auto pdf_bsdf = path_states.read_pdf_bsdf(path_id);
            auto beta = path_states.read_beta(path_id);
            auto eval = light_sampler()->evaluate_miss(wi, swl, time);
            auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
            auto Li = beta * eval.L * mis_weight;
            auto pixel_id = path_states.read_pixel_index(path_id);
            auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
            camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
        }
        auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
        if (!gathering)
            invalid_queue.write(queue_id, path_id);
        
    });

    LUISA_INFO("Compiling light evaluation kernel.");
    auto evaluate_light_shader = device.compile_async<1>([&](BufferUInt light_indices,
                                                             BufferUInt sample_queue, BufferUInt sample_queue_size,
                                                             BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = light_indices.read(dispatch_id);
        if (!pipeline().lights().empty()) {
            auto ray = path_states.read_ray(path_id);
            auto hit = path_states.read_hit(path_id);
            auto [u_wl, swl] = path_states.read_swl(path_id);
            auto pdf_bsdf = path_states.read_pdf_bsdf(path_id);
            auto beta = path_states.read_beta(path_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
            auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
            auto Li = beta * eval.L * mis_weight;
            auto pixel_id = path_states.read_pixel_index(path_id);
            auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
            camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
            auto shape = pipeline().geometry()->instance(hit.inst);
            $if(shape.has_surface()) {
                auto queue_id = sample_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    sample_queue.write(queue_id, path_id);
            }
            $else {
                auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    invalid_queue.write(queue_id, path_id);
            };
        } else {
            auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
            if (!gathering)
                invalid_queue.write(queue_id, path_id);
        }
    });

    LUISA_INFO("Compiling light sampling kernel.");
    auto sample_light_shader = device.compile_async<1>([&](BufferUInt sample_indices, 
                                                           BufferUInt surface_queue, BufferUInt surface_queue_size,
                                                           BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = sample_indices.read(dispatch_id);
        sampler()->load_state(path_id);
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        sampler()->save_state(path_id);
        auto ray = path_states.read_ray(path_id);
        auto hit = path_states.read_hit(path_id);
        auto it = pipeline().geometry()->interaction(ray, hit);
        auto [u_wl, swl] = path_states.read_swl(path_id);
        auto light_sample = light_sampler()->sample(
            *it, u_light_selection, u_light_surface, swl, time);
        // trace shadow ray
        auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);//if occluded, transit to invalid
        light_samples.write_emission(path_id, ite(occluded, 0.f, 1.f) * light_sample.eval.L);
        light_samples.write_wi_and_pdf(path_id, light_sample.shadow_ray->direction(),
                                       ite(occluded, 0.f, light_sample.eval.pdf));
        auto queue_id = surface_queue_size.atomic(0u).fetch_add(1u);
        if (!gathering)
            surface_queue.write(queue_id, path_id);

    });

    LUISA_INFO("Compiling surface evaluation kernel.");
    auto evaluate_surface_shader = device.compile_async<1>([&](BufferUInt surface_indices, 
                                                               BufferUInt intersect_queue, BufferUInt intersect_queue_size,
                                                               BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
            auto path_id = surface_indices.read(dispatch_id);
            sampler()->load_state(path_id);
            auto depth = path_states.read_depth(path_id);
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<WavefrontPathTracingv2>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            sampler()->save_state(path_id);
            auto ray = path_states.read_ray(path_id);
            auto hit = path_states.read_hit(path_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto u_wl_and_swl = path_states.read_swl(path_id);
            auto &&u_wl = u_wl_and_swl.first;
            auto &&swl = u_wl_and_swl.second;
            auto beta = path_states.read_beta(path_id);
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            auto wo = -ray->direction();
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(it, swl, wo, 1.f, time);

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
                    auto light_wi_and_pdf = light_samples.read_wi_and_pdf(path_id);
                    auto pdf_light = light_wi_and_pdf.w;
                    $if(light_wi_and_pdf.w > 0.f) {
                        auto eval = closure->evaluate(wo, light_wi_and_pdf.xyz());
                        auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                        // update Li
                        auto Ld = light_samples.read_emission(path_id);
                        auto Li = mis_weight / pdf_light * beta * eval.f * Ld;
                        auto pixel_id = path_states.read_pixel_index(path_id);
                        auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                        
                        camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
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
                auto rr_threshold = node<WavefrontPathTracingv2>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, 0.05f);
                $if(depth + 1u >= rr_depth) {
                    terminated = q < rr_threshold & u_rr >= q;
                    beta *= ite(q < rr_threshold, 1.f / q, 1.f);
                };
            };
            $if(depth+1 >= node<WavefrontPathTracingv2>()->max_depth()) {
                terminated = true;
            };
            auto pixel_id = path_states.read_pixel_index(path_id);
            auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
            Float termi = 0.f;
            $if(terminated) {
                termi=1.f;
            };

            $if(!terminated) {
                path_states.write_depth(path_id, depth + 1);
                path_states.write_beta(path_id, beta);
                path_states.write_ray(path_id, ray);
                auto queue_id = intersect_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    intersect_queue.write(queue_id, path_id);
            }
            $else {
                auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    invalid_queue.write(queue_id, path_id);
            };
    });

    LUISA_INFO("Compiling initializtation kernel.");
    auto mark_invalid_shader = device.compile_async<1>([&](
                                                           BufferUInt invalid_queue, BufferUInt invalid_queue_size) noexcept {
        auto dispatch_id = dispatch_x();
        if (!gathering)
            invalid_queue.write(dispatch_id, dispatch_id);
        invalid_queue_size.write(0u,state_count);
        
    });
    auto gathering_shader = device.compile_async<1>([&](
                                                        BufferUInt queue, BufferUInt queue_size) noexcept {
        if (gathering) {
            auto dispatch_id = dispatch_x();
            //TODO
        }
    });
    auto test_shader = device.compile_async<1>([&](BufferUInt queue, UInt queue_size,
                                                   BufferUInt queue_out1, BufferUInt queue_out1_size,
                                                   BufferUInt queue_out2, BufferUInt queue_out2_size,
                                                   UInt kernel_num) noexcept {
        
        auto dispatch_id = dispatch_x();
        auto size = queue_size;
        $if(dispatch_id < size) {
        //$if(true){
            auto path_id = queue.read(dispatch_id);
            auto pixel_id = (dispatch_id) % pixel_count;
            auto sample_id = dispatch_id / pixel_count;
            auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
            Float u_test = 0.0f;
            $if(kernel_num == 0) {
                sampler()->start(pixel_coord, sample_id);
                auto u_filter = sampler()->generate_pixel_2d();
                auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
                u_test = sampler()->generate_1d();
                sampler()->save_state(path_id);
                auto camera_sample = camera->generate_ray(pixel_coord, 0.0f, u_filter, u_lens);
                path_states.write_ray(path_id, camera_sample.ray);
            }
            $else {
                sampler()->load_state(path_id);
                auto ray = path_states.read_ray(path_id);
                //camera->film()->accumulate(pixel_coord, ray->direction(), 1.f);
                u_test = sampler()->generate_1d();
                sampler()->save_state(path_id);
                //u_test = (dispatch_id % 11) * 0.1f;
            };
            $if(u_test<0.9f) {
                auto queue_id = queue_out1_size.atomic(0u).fetch_add(1u);
                queue_out1.write(queue_id, path_id);
            }
            $else {
                auto queue_id = queue_out2_size.atomic(0u).fetch_add(1u);
                queue_out2.write(queue_id, path_id);
            };
            


        };
    });
    // wait for the compilation of all shaders
    generate_rays_shader.wait();
    intersect_shader.wait();
    evaluate_miss_shader.wait();
    evaluate_surface_shader.wait();
    evaluate_light_shader.wait();
    sample_light_shader.wait();
    mark_invalid_shader.wait();
    gathering_shader.wait();
    test_shader.wait();
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);

    LUISA_INFO("Rendering started.");
    // create path states
    
    auto shutter_samples = camera->node()->shutter_samples();

 
    Clock clock;
    ProgressBar progress_bar;
    progress_bar.update(0.0);
    auto launch_limit = state_count / (KERNEL_COUNT - 1);
    int shutter_spp = 0;
    for (auto s : shutter_samples) {

        shutter_spp += s.spp;
        auto time = s.point.time;
        pipeline().update(command_buffer, time);
        for (int i = 0u; i < KERNEL_COUNT; ++i) {
            queues[i].clear_counter_buffer(command_buffer);
        }
        auto launch_state_count=s.spp * pixel_count;
        auto last_committed_state = launch_state_count;
        auto queues_empty = true;
        command_buffer << mark_invalid_shader.get()(queues[INVALID].index_buffer(command_buffer), queues[INVALID].counter_buffer(command_buffer)).dispatch(state_count);
        auto iteration = 0;


        //test case
        /*
        const uint test_iteration = 590;
        for (auto i = 0u; i < test_iteration; ++i) {
            //command_buffer << pipeline().printer().retrieve();

            for (auto i = 0u; i < KERNEL_COUNT; ++i) {
                queues[i].catch_counter(command_buffer);
            }
            command_buffer << synchronize();//catch the queue counters
            auto max_count = 0u;
            auto max_index = -1;
            for (auto i = 0u; i < KERNEL_COUNT; ++i) {
                if (queues[i].host_counter() > 0) {
                    
                    if (queues[i].host_counter() > max_count) {
                        max_count = queues[i].host_counter();
                        max_index = i;
                    }
                }
            }
            LUISA_ASSERT(max_index != -1, "no path found error");
            auto test1 = max_index % KERNEL_COUNT;
            auto test2 = (max_index + 1) % KERNEL_COUNT;
            auto test3 = (max_index + 2) % KERNEL_COUNT;
            queues[test1].clear_counter_buffer(command_buffer);
            LUISA_INFO("Launching test kernel {} with size {}", test1, queues[test1].host_counter());
            command_buffer << test_shader.get()(queues[test1].index_buffer(command_buffer), queues[test1].host_counter(),
                                                queues[test2].index_buffer(command_buffer), queues[test2].counter_buffer(command_buffer),
                                                queues[test3].index_buffer(command_buffer), queues[test3].counter_buffer(command_buffer),
                                                (uint)test1).dispatch(state_count);

        }
        */


        
        while (launch_state_count > 0 || !queues_empty) {
            iteration += 1;
            //command_buffer << pipeline().printer().retrieve();
            queues_empty = true;
            for (auto i = 0u; i < KERNEL_COUNT; ++i) {
                queues[i].catch_counter(command_buffer);
            }
            command_buffer << synchronize();//catch the queue counters

            for (auto i = 0u; i < KERNEL_COUNT; ++i) {
                //LUISA_INFO("kernel {} has size {}", i, queues[i].host_counter());
            }
            if (queues[INVALID].host_counter() > state_count / 2&&launch_state_count>0) {//launch new kernel
                auto path_indices = queues[INVALID].index_buffer(command_buffer);
                auto intersect_indices = queues[INTERSECT].index_buffer(command_buffer);
                auto intersect_size = queues[INTERSECT].counter_buffer(command_buffer);
                queues[INVALID].clear_counter_buffer(command_buffer);
                auto generate_count = std::min(launch_state_count, queues[INVALID].host_counter());
                command_buffer << generate_rays_shader.get()(path_indices, intersect_indices,intersect_size,
                                                shutter_spp-s.spp,s.spp*pixel_count-launch_state_count, time,s.point.weight)
                                      .dispatch(generate_count);
                launch_state_count -= generate_count;
                queues_empty = false;
                //LUISA_INFO("Generate new kernel size {}", generate_count);
                continue;
            }
            auto max_count = 0u;
            auto max_index = -1;

            
            for (auto i = 1u; i < KERNEL_COUNT; ++i) {
                //LUISA_INFO("kernel queue {} has size {}", KernelName[i], queues[i].host_counter());
                if (queues[i].host_counter() > 0) {
                    queues_empty = false;
                    if (queues[i].host_counter() > max_count) {
                        max_count = queues[i].host_counter();
                        max_index = i;
                    }
                }
            }
            if (max_index != -1) {
                queues[max_index].clear_counter_buffer(command_buffer);
                //LUISA_INFO("Launch kernel {} for size {}", KernelName[max_index], queues[max_index].host_counter());
                switch (max_index) {
                    case INTERSECT:
                        if (gathering) {
                            command_buffer << gathering_shader.get()(queues[INTERSECT].index_buffer(command_buffer),
                                queues[INTERSECT].counter_buffer(command_buffer)).dispatch(state_count);
                        }
                        command_buffer << intersect_shader.get()(queues[INTERSECT].index_buffer(command_buffer),
                                                                 queues[SAMPLE].index_buffer(command_buffer), queues[SAMPLE].counter_buffer(command_buffer),
                                                                 queues[LIGHT].index_buffer(command_buffer), queues[LIGHT].counter_buffer(command_buffer),
                                                                 queues[MISS].index_buffer(command_buffer), queues[MISS].counter_buffer(command_buffer),
                                                                 queues[INVALID].index_buffer(command_buffer), queues[INVALID].counter_buffer(command_buffer))
                                              .dispatch(queues[INTERSECT].host_counter());
                        break;
                    case MISS:
                        command_buffer << evaluate_miss_shader.get()(queues[MISS].index_buffer(command_buffer),
                                                                     queues[INVALID].index_buffer(command_buffer), queues[INVALID].counter_buffer(command_buffer), time)
                                              .dispatch(queues[MISS].host_counter());
                        break;
                    case LIGHT:
                        command_buffer << evaluate_light_shader.get()(queues[LIGHT].index_buffer(command_buffer),
                                                                      queues[SAMPLE].index_buffer(command_buffer), queues[SAMPLE].counter_buffer(command_buffer),
                                                                      queues[INVALID].index_buffer(command_buffer), queues[INVALID].counter_buffer(command_buffer), time)
                                              .dispatch(queues[LIGHT].host_counter());
                        break;
                    case SAMPLE:
                        command_buffer << sample_light_shader.get()(queues[SAMPLE].index_buffer(command_buffer),
                                                                    queues[SURFACE].index_buffer(command_buffer), queues[SURFACE].counter_buffer(command_buffer),
                                                                    queues[INVALID].index_buffer(command_buffer), queues[INVALID].counter_buffer(command_buffer), time)
                                              .dispatch(queues[SAMPLE].host_counter());
                        break;
                    case SURFACE:
                        command_buffer << evaluate_surface_shader.get()(queues[SURFACE].index_buffer(command_buffer),
                                                                        queues[INTERSECT].index_buffer(command_buffer), queues[INTERSECT].counter_buffer(command_buffer),
                                                                        queues[INVALID].index_buffer(command_buffer), queues[INVALID].counter_buffer(command_buffer), time)
                                              .dispatch(queues[SURFACE].host_counter());
                        break;
                }
            }
            auto launches_per_commit =
                display() && !display()->should_close() ?
                    node<WavefrontPathTracingv2>()->display_interval() :
                    16u;
            if (last_committed_state-launch_state_count >= launches_per_commit*pixel_count) {
                last_committed_state = launch_state_count;
                auto p = (shutter_spp-last_committed_state/static_cast<double>(pixel_count)) / static_cast<double>(spp);
                if (display() && display()->update(command_buffer, static_cast<uint>(shutter_spp - last_committed_state / static_cast<double>(pixel_count)))) {
                    progress_bar.update(p);
                } else {
                    command_buffer << [p, &progress_bar] { progress_bar.update(p); };
                }
            }
        }
        
        LUISA_INFO("Total iteration {}", iteration);
        command_buffer << pipeline().printer().retrieve();
        
    }
    command_buffer << synchronize();
    progress_bar.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WavefrontPathTracingv2)
