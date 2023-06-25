//
// Created by Mike Smith on 2022/1/10.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/thread_pool.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <dsl/syntax.h>

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
    ShaderOption o{};
    o.enable_debug_info = true;
    return global_thread_pool().async([&device, o, kernel] {
        return device.compile(kernel, o);
    });
}
class WavefrontPathTracingv2 final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _state_limit;
    bool _gathering;
    bool _test_case;
    bool _compact;
    bool _use_tag_sort;

public:
    WavefrontPathTracingv2(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _state_limit{std::max(desc->property_uint_or_default("state_limit", 1024 * 1024 * 32u), 1u)},
          _gathering{desc->property_bool_or_default("gathering", true)},
          _use_tag_sort{desc->property_bool_or_default("use_tag_sort", true)},
          _test_case{desc->property_bool_or_default("test_case", false)},
          _compact{desc->property_bool_or_default("compact", true)} {}

    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto use_tag_sort() const noexcept { return _use_tag_sort; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto state_limit() const noexcept { return _state_limit; }
    [[nodiscard]] auto gathering() const noexcept { return _gathering; }
    [[nodiscard]] auto test_case() const noexcept { return _test_case; }
    [[nodiscard]] auto compact() const noexcept { return _compact; }
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
    Buffer<uint> _kernel_index;
    Buffer<uint> _depth;
    Buffer<uint> _pixel_index;
    //Buffer<uint> _kernel_count;
    luisa::vector<uint> _host_count;
    Buffer<Ray> _ray;
    Buffer<Hit> _hit;
    bool _gathering;

public:
    PathStateSOA(const Spectrum::Instance *spectrum, size_t size, bool gathering) noexcept
        : _spectrum{spectrum} {
        auto &&device = spectrum->pipeline().device();
        auto dimension = spectrum->node()->dimension();
        _beta = device.create_buffer<float>(size * dimension);
        _pdf_bsdf = device.create_buffer<float>(size);
        _gathering = gathering;
        if (_gathering)
            _kernel_index = device.create_buffer<uint>(size);
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
            s[i] = _beta->read(offset + i);
        }
        return s;
    }
    [[nodiscard]] auto read_kernel_index(Expr<uint> index) const noexcept {
        return _kernel_index->read(index);
    }
    void write_kernel_index(Expr<uint> index, Expr<uint> kernel_index) noexcept {
        _kernel_index->write(index, kernel_index);
    }
    [[nodiscard]] auto read_ray(Expr<uint> index) const noexcept {
        return _ray->read(index);
    }
    [[nodiscard]] auto read_hit(Expr<uint> index) const noexcept {
        return _hit->read(index);
    }
    void write_ray(Expr<uint> index, Expr<Ray> ray) noexcept {
        _ray->write(index, ray);
    }
    void write_hit(Expr<uint> index, Expr<Hit> hit) noexcept {
        _hit->write(index, hit);
    }
    [[nodiscard]] auto read_depth(Expr<uint> index) const noexcept {
        return _depth->read(index);
    }
    [[nodiscard]] auto read_pixel_index(Expr<uint> index) const noexcept {
        return _pixel_index->read(index);
    }
    void write_pixel_index(Expr<uint> index, Expr<uint> pixel_index) noexcept {
        _pixel_index->write(index, pixel_index);
    }
    void write_depth(Expr<uint> index, Expr<uint> depth) noexcept {
        _depth->write(index, depth);
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
    auto read_wavelength_sample(Expr<uint> index) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            return _wl_sample->read(index);
        } else {
            return def(0.f);
        }
    }
    void terminate_secondary_wavelengths(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample->write(index, -u_wl);
        }
    }

    [[nodiscard]] auto read_pdf_bsdf(Expr<uint> index) const noexcept {
        return _pdf_bsdf->read(index);
    }
    void write_pdf_bsdf(Expr<uint> index, Expr<float> pdf) noexcept {
        _pdf_bsdf->write(index, pdf);
    }
#define MOVE(entry, from, to)           \
    {                                   \
        auto inst = read_##entry(from); \
        write_##entry(to, inst);        \
    }
    void move(Expr<uint> from, Expr<uint> to) noexcept {
        MOVE(beta, from, to);
        MOVE(pdf_bsdf, from, to);
        MOVE(ray, from, to);
        MOVE(hit, from, to);
        MOVE(depth, from, to);
        MOVE(pixel_index, from, to);
        if (_gathering) {
            MOVE(kernel_index, from, to);
        }
        if (!_spectrum->node()->is_fixed()) {
            MOVE(wavelength_sample, from, to);
        }
    }
#undef MOVE
};

class LightSampleSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _emission;
    Buffer<float4> _wi_and_pdf;
    Buffer<uint> _surface_tag;
    Buffer<uint> _tag_counter;
    bool _use_tag_sort;

public:
    LightSampleSOA(const Spectrum::Instance *spec, size_t size, size_t tag_size) noexcept
        : _spectrum{spec} {
        auto &&device = spec->pipeline().device();
        auto dimension = spec->node()->dimension();
        _emission = device.create_buffer<float>(size * dimension);
        _wi_and_pdf = device.create_buffer<float4>(size);
        if (tag_size > 0) {
            _use_tag_sort = true;
            _surface_tag = device.create_buffer<uint>(size);
            _tag_counter = device.create_buffer<uint>(tag_size);
        } else {
            _use_tag_sort = false;
            _surface_tag = device.create_buffer<uint>(1u);
            _tag_counter = device.create_buffer<uint>(1u);
        }
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
    [[nodiscard]] auto read_surface_tag(Expr<uint> index) const noexcept {
        return _surface_tag->read(index);
    }
    void write_surface_tag(Expr<uint> index, Expr<uint> tag) noexcept {
        _surface_tag->write(index, tag);
    }
    void increase_tag(Expr<uint> index) noexcept {
        _tag_counter->atomic(index).fetch_add(1u);
    }
    [[nodiscard]] BufferView<uint> tag_counter() const noexcept {
        return _tag_counter;
    }
    [[nodiscard]] BufferView<uint> surface_tag() const noexcept {
        return _surface_tag;
    }
#define MOVE(entry, from, to)           \
    {                                   \
        auto inst = read_##entry(from); \
        write_##entry(to, inst);        \
    }
    void move(Expr<uint> from, Expr<uint> to) noexcept {
        MOVE(emission, from, to);
        if (_use_tag_sort) {
            MOVE(surface_tag, from, to);
        }
        auto inst = read_wi_and_pdf(from);
        write_wi_and_pdf(to, inst.xyz(), inst.w);
    }
#undef MOVE
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
            _counter_buffer->write(dispatch_x(), 0u);
        });
        _host_counter = 0;
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
    [[nodiscard]] uint host_counter() const noexcept {
        return _host_counter;
    }
    void catch_counter(CommandBuffer &command_buffer) noexcept {
        command_buffer << _counter_buffer.view(0, 1).copy_to(&_host_counter);
    }
};

class AggregatedRayQueue {
private:
    Buffer<uint> _index_buffer;
    Buffer<uint> _counter_buffer;
    Shader1D<> _clear_counters;
    uint _kernel_count;
    luisa::vector<uint> _host_counter;
    luisa::vector<uint> _offsets;
    bool _gathering;
    size_t _size;
public:
    AggregatedRayQueue(Device &device, size_t size, uint kernel_count,bool gathering) noexcept
        : _index_buffer{device.create_buffer<uint>(gathering?size:kernel_count*size)},
          _counter_buffer{device.create_buffer<uint>(kernel_count)},
          _kernel_count{kernel_count},
          _size{size},
          _gathering{gathering}{
        _host_counter.resize(kernel_count);
        _offsets.resize(kernel_count);
        _clear_counters = device.compile<1>([this] {
            _counter_buffer->write(dispatch_x(), 0u);
        });
    }
    void clear_counter_buffer(CommandBuffer &command_buffer,int index=-1) noexcept {
        //if (_current_counter == counter_buffer_size-1) {
        //   _current_counter = 0u;
        if(index==-1){
            command_buffer << _clear_counters().dispatch(_kernel_count);
        }
        else{
            uint zero=0u;
            command_buffer << counter_buffer(index).copy_from(&zero);
        }
        //} else
        //    _current_counter++;
    }
    [[nodiscard]] BufferView<uint> counter_buffer(uint index) noexcept {
        return _counter_buffer.view(index,1);
    }
    [[nodiscard]] BufferView<uint> index_buffer(uint index) noexcept {
        if(_gathering)
            return _index_buffer.view(_offsets[index],_host_counter[index]);
        else return _index_buffer.view(index*_size,_size);
    }
    [[nodiscard]] uint host_counter(uint index) const noexcept {
        return _host_counter[index];
    }
    void catch_counter(CommandBuffer &command_buffer) noexcept {
        command_buffer << _counter_buffer.view(0, _kernel_count).copy_to(_host_counter.data());
        command_buffer <<synchronize();
        uint prev=0u;
        for(auto i=0u;i<_kernel_count;++i){
            uint now=_host_counter[i];
            _offsets[i]=prev;
            prev+=now;
        }
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
void push_if(Expr<bool> pred, Expr<uint> value, Expr<Buffer<uint>> buffer, Expr<Buffer<uint>> counter, bool gathering) noexcept {
    Shared<uint> index{1};
    $if(thread_x() == 0u) { index.write(0u, 0u); };
    sync_block();
    auto local_index = def(0u);
    $if(pred) { local_index = index.atomic(0).fetch_add(1u); };
    sync_block();
    $if(thread_x() == 0u) {
        auto local_count = index.read(0u);
        auto global_offset = counter.atomic(0u).fetch_add(local_count);
        index.write(0u, global_offset);
    };
    sync_block();
    $if(pred) {
        auto global_index = index.read(0u) + local_index;
        if (!gathering) {
            buffer.write(global_index, value);
        }
    };
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
    auto test_case = node<WavefrontPathTracingv2>()->test_case();
    auto compact = node<WavefrontPathTracingv2>()->compact();
    auto use_tag_sort = node<WavefrontPathTracingv2>()->use_tag_sort();
    bool use_sort = true;
    bool direct_launch = false;
    LUISA_INFO("Wavefront path tracing configurations: "
               "resolution = {}x{}, spp = {}, state_count = {}.",
               resolution.x, resolution.y, spp, state_count);

    auto spectrum = pipeline().spectrum();
    PathStateSOA path_states{spectrum, state_count, gathering};
    LightSampleSOA light_samples{spectrum, state_count, use_tag_sort ? pipeline().surfaces().size() : 0};
    sampler()->reset(command_buffer, resolution, state_count, spp);
    command_buffer << synchronize();
    AggregatedRayQueue aqueue{device,state_count,KERNEL_COUNT,gathering};
    //RayQueue queues[KERNEL_COUNT] = {{device, state_count}, {device, state_count}, {device, state_count}, {device, state_count}, {device, state_count}, {device, state_count}};
    RayQueue empty_queue{device, state_count};
    LUISA_INFO("Compiling ray generation kernel.");
    Clock clock_compile;
    auto generate_rays_shader = compile_async<1>(device,[&](BufferUInt path_indices, UInt offset, BufferUInt intersect_indices, BufferUInt intersect_size,
                                                            UInt base_spp, UInt extra_sample_id, Float time, Float shutter_weight, UInt n) noexcept {
        auto path_id = def(0u);
        auto pixel_coord = def(make_uint2(0u));

        $if (dispatch_x() < n) {

            auto dispatch_id = dispatch_x();
            auto pixel_id = (extra_sample_id + dispatch_id) % pixel_count;
            auto sample_id = base_spp + (extra_sample_id + dispatch_id) / pixel_count;
            pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
            camera->film()->accumulate(pixel_coord, make_float3(0.f), 1.f);

            if (compact) {
                if (use_sort)
                    path_id = dispatch_id;
                else
                    path_id = offset + dispatch_id;
            } else {
                path_id = path_indices.read(dispatch_id);
            }
            //$if(path_id < offset) {
            //    pipeline().printer().info("path_id {}, offset {}", path_id,offset);
            //};
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
        };

        // TODO: this could be entirely optimized out
        auto queue_id = def(0u);
        {
             Shared<uint> index{1u};
             $if(thread_x() == 0u) { index.write(0u, 0u); };
             sync_block();
             auto local_index = def(0u);
             $if (dispatch_x() < n) {
                 local_index = index.atomic(0).fetch_add(1u);
             };
             sync_block();
             $if(thread_x() == 0u) {
                 auto local_count = index.read(0u);
                 auto global_offset = intersect_size.atomic(0u).fetch_add(local_count);
                 index.write(0u, global_offset);
             };
             sync_block();
             queue_id = index.read(0u) + local_index;
        }

        $if (dispatch_x() < n) {
             if (!gathering)
                 intersect_indices.write(queue_id, path_id);
             else {
                 path_states.write_kernel_index(path_id, (uint)INTERSECT);
             }
        };
    });

    LUISA_INFO("Compiling intersection kernel.");
    auto intersect_shader = compile_async<1>(device, [&](BufferUInt intersect_indices, BufferUInt surface_queue, BufferUInt surface_queue_size,
                                                         BufferUInt light_queue, BufferUInt light_queue_size,
                                                         BufferUInt escape_queue, BufferUInt escape_queue_size,
                                                         BufferUInt invalid_queue, BufferUInt invalid_queue_size) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = intersect_indices.read(dispatch_id);
        Bool condition = true;
        if (direct_launch) {
            path_id = dispatch_id;
            auto kernel_index = path_states.read_kernel_index(path_id);
            condition = (kernel_index == (uint)INTERSECT);
        }
        $if(condition) {
            auto ray = path_states.read_ray(path_id);
            auto hit = pipeline().geometry()->trace_closest(ray);
            path_states.write_hit(path_id, hit);
            $if(!hit->miss()) {
                auto shape = pipeline().geometry()->instance(hit.inst);

                $if(shape.has_light()) {
                    auto queue_id = light_queue_size.atomic(0u).fetch_add(1u);
                    if (!gathering)
                        light_queue.write(queue_id, path_id);
                    else
                        path_states.write_kernel_index(path_id, (uint)LIGHT);
                }
                $else {
                    $if(shape.has_surface()) {
                        auto queue_id = surface_queue_size.atomic(0u).fetch_add(1u);
                        if (!gathering)
                            surface_queue.write(queue_id, path_id);
                        else
                            path_states.write_kernel_index(path_id, (uint)SAMPLE);
                    }
                    $else {
                        auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                        if (!gathering)
                            invalid_queue.write(queue_id, path_id);
                        else
                            path_states.write_kernel_index(path_id, (uint)INVALID);
                    };
                };
            }
            $else {
                if (pipeline().environment()) {
                    auto queue_id = escape_queue_size.atomic(0u).fetch_add(1u);
                    if (!gathering)
                        escape_queue.write(queue_id, path_id);
                    else
                        path_states.write_kernel_index(path_id, (uint)MISS);
                } else {
                    auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                    if (!gathering)
                        invalid_queue.write(queue_id, path_id);
                    else
                        path_states.write_kernel_index(path_id, (uint)INVALID);
                }
            };
        };
    });

    LUISA_INFO("Compiling environment evaluation kernel.");
    auto evaluate_miss_shader = compile_async<1>(device, [&](BufferUInt miss_indices,
                                                             BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = miss_indices.read(dispatch_id);
        Bool condition = true;
        if (direct_launch) {
            path_id = dispatch_id;
            auto kernel_index = path_states.read_kernel_index(path_id);
            condition = (kernel_index == (uint)MISS);
        }
        $if(condition) {
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
            else
                path_states.write_kernel_index(path_id, (uint)INVALID);
        };
    });

    LUISA_INFO("Compiling light evaluation kernel.");
    auto evaluate_light_shader = compile_async<1>(device, [&](BufferUInt light_indices,
                                                              BufferUInt sample_queue, BufferUInt sample_queue_size,
                                                              BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = light_indices.read(dispatch_id);
        Bool condition = true;
        if (direct_launch) {
            path_id = dispatch_id;
            auto kernel_index = path_states.read_kernel_index(path_id);
            condition = (kernel_index == (uint)LIGHT);
        }
        $if(condition) {

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
                    else
                        path_states.write_kernel_index(path_id, (uint)SAMPLE);
                }
                $else {
                    auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                    if (!gathering)
                        invalid_queue.write(queue_id, path_id);
                    else
                        path_states.write_kernel_index(path_id, (uint)INVALID);
                };
            } else {
                auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
                if (!gathering)
                    invalid_queue.write(queue_id, path_id);
                else
                    path_states.write_kernel_index(path_id, (uint)INVALID);
            }
        };
    });
    LUISA_INFO("Compiling light sampling kernel.");
    auto sample_light_shader = compile_async<1>(device,[&](BufferUInt sample_indices,
                                                           BufferUInt surface_queue, BufferUInt surface_queue_size,
                                                           BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = sample_indices.read(dispatch_id);
        Bool condition = true;
        if(direct_launch) {
            path_id = dispatch_id;
            auto kernel_index = path_states.read_kernel_index(path_id);
            condition = (kernel_index == (uint)SAMPLE);
        }
        $if(condition) {
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
        if (use_tag_sort) {
            auto surface_tag = it->shape().surface_tag();
            light_samples.write_surface_tag(path_id, surface_tag);
            light_samples.increase_tag(surface_tag);
        }
        auto queue_id = surface_queue_size.atomic(0u).fetch_add(1u);
        if (!gathering)
            surface_queue.write(queue_id, path_id);
        else
            path_states.write_kernel_index(path_id, (uint)SURFACE);
        };

    });

    LUISA_INFO("Compiling surface evaluation kernel.");
    auto evaluate_surface_shader = compile_async<1>(device,[&](BufferUInt surface_indices,
                                                               BufferUInt intersect_queue, BufferUInt intersect_queue_size,
                                                               BufferUInt invalid_queue, BufferUInt invalid_queue_size, Float time) noexcept {
        auto dispatch_id = dispatch_x();
        auto path_id = surface_indices.read(dispatch_id);
        Bool condition=true;
        if(direct_launch){
            path_id = dispatch_id;
            auto kernel_index = path_states.read_kernel_index(path_id);
            condition = (kernel_index == (uint)SURFACE);
        }
        $if(condition){
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
            else
                path_states.write_kernel_index(path_id, (uint)INTERSECT);
        }
        $else {
            auto queue_id = invalid_queue_size.atomic(0u).fetch_add(1u);
            if (!gathering)
                invalid_queue.write(queue_id, path_id);
            else
                path_states.write_kernel_index(path_id, (uint)INVALID);
        };
        };
    });

    LUISA_INFO("Compiling management kernels.");
    auto mark_invalid_shader = compile_async<1>(device, [&](
                                                            BufferUInt invalid_queue, BufferUInt invalid_queue_size) noexcept {
        auto dispatch_id = dispatch_x();
        if (!gathering)
            invalid_queue.write(dispatch_id, dispatch_id);
        invalid_queue_size.write(0u, state_count);
        if (gathering)
            path_states.write_kernel_index(dispatch_id, 0u);

    });
    auto gather_shader = compile_async<1>(device,[&](BufferUInt queue, BufferUInt queue_size,
                                                     UInt kernel_id, UInt n) noexcept {
        if (gathering) {
            auto path_id = dispatch_x();
            auto kernel = def(0u);
            $if (dispatch_x() < n) {
                kernel = path_states.read_kernel_index(path_id);
            };
            /*$if(kernel == kernel_id) {
                auto slot = queue_size.atomic(0u).fetch_add(1u);
                                queue.write(slot, path_id);
            };*/
            
            auto slot = def(0u);
            {
                Shared<uint> index{1u};
                $if (thread_x() == 0u) { index.write(0u, 0u); };
                sync_block();
                auto local_index = def(0u);
                $if (dispatch_x() < n & kernel == kernel_id) {
                    local_index = index.atomic(0u).fetch_add(1u);
                };
                sync_block();
                $if (thread_x() == 0u) {
                    auto local_count = index.read(0u);
                    auto global_offset = queue_size.atomic(0u).fetch_add(local_count);
                    index.write(0u, global_offset);
                };
                sync_block();
                slot = index.read(0u) + local_index;
            }
            $if(dispatch_x() < n & kernel == kernel_id) {
                queue.write(slot, path_id);
            };

        }
    });
    auto sort_tag_gather_shader = compile_async<1>(device, [&](BufferUInt queue, BufferUInt tags, BufferUInt tag_counter, UInt kernel_id, UInt tag_size) noexcept {
        if (gathering && use_tag_sort) {
            auto path_id = dispatch_x();
            $if(path_id < state_count) {
                auto kernel = path_states.read_kernel_index(path_id);
                auto tag = tags.read(path_id);
                $if(kernel == kernel_id) {
                    if (pipeline().surfaces().size() <= 32) {//not sure what is the proper threshold

                        for (auto i = 0u; i < pipeline().surfaces().size(); ++i) {
                            $if(tag == i) {
                                auto queue_id = tag_counter.atomic(i).fetch_add(1u);
                                queue.write(queue_id, path_id);
                            };
                        }
                    } else {
                        auto queue_id = tag_counter.atomic(tag).fetch_add(1u);
                        queue.write(queue_id, path_id);
                    }
                };
            };
        }
    });
    auto bucket_update_shader = compile_async<1>(device, [&](BufferUInt tag_counter, UInt tag_size) noexcept {
        if (use_tag_sort) {
            UInt prev = 0u;
            $for(i, 0u, tag_size) {

                UInt now = tag_counter.read(i);
                tag_counter.write(i, prev);
                prev += now;
            };
        }
    });
    auto bucket_reset_shader = compile_async<1>(device, [&](BufferUInt tag_counter) noexcept {
        if (use_tag_sort)
            tag_counter.write(dispatch_x(), 0u);
    });
    auto compact_shader = compile_async<1>(device, [&](UInt move_offset, BufferUInt invalid_queue, BufferUInt invalid_counter, BufferUInt queue, BufferUInt queue_size) noexcept {
        if (compact) {
            auto dispatch_id = dispatch_x();
            auto size = queue_size.read(0u);
            auto path_id = queue.read(dispatch_id);
            $if((dispatch_id < size) & (path_id >= move_offset)) {
                auto slot = invalid_counter.atomic(0u).fetch_add(1u);
                auto new_id = invalid_queue.read(slot);
                path_states.move(path_id, new_id);
                if (gathering) {
                    auto kernel = path_states.read_kernel_index(path_id);
                    $if(kernel == (uint)SURFACE) {
                        light_samples.move(path_id, new_id);
                    };
                } else {
                    light_samples.move(path_id, new_id);
                }
                sampler()->load_state(path_id);
                sampler()->save_state(new_id);
                queue.write(dispatch_id, new_id);
                if (gathering)
                    path_states.write_kernel_index(path_id, (uint)INVALID);//in the end, the generation could left some states unchanged
                //pipeline().printer().info("move {} to {}",path_id, new_id);
            };
        }
    });

    auto ordering_shader = compile_async<1>(device, [&](UInt move_offset, BufferUInt queue, UInt queue_size) noexcept{
        if (compact) {
            auto dispatch_id = dispatch_x();
            auto size = queue_size;
            auto path_id = queue.read(dispatch_id);
            $if((dispatch_id < size)) {
                auto new_id = move_offset + dispatch_id;
                auto state_test = path_states.read_kernel_index(new_id);
                path_states.move(path_id, new_id);
                if (gathering) {
                    auto kernel = path_states.read_kernel_index(path_id);
                    $if(kernel == (uint)SURFACE) {
                        light_samples.move(path_id, new_id);
                    };
                } else {
                    light_samples.move(path_id, new_id);
                }
                sampler()->load_state(path_id);
                sampler()->save_state(new_id);
                if (!gathering)
                    queue.write(dispatch_id, new_id);
                else
                    path_states.write_kernel_index(path_id, (uint)INVALID);//in the end, the generation could left some states unchanged
                //pipeline().printer().info("move {} to {}",path_id, new_id);
            };
        }
    });
    auto empty_gather_shader = compile_async<1>(device, [&](UInt move_offset, BufferUInt invalid_queue, UInt invalid_queue_size, BufferUInt queue, BufferUInt queue_size) noexcept {
        auto dispatch_id = dispatch_x();
        auto size = invalid_queue_size;
        auto path_id = invalid_queue.read(dispatch_id);
        $if((dispatch_id < size) & (path_id < move_offset)) {
            auto queue_id = queue_size.atomic(0u).fetch_add(1u);
            queue.write(queue_id, path_id);
            //pipeline().printer().info("{} is a slot", path_id);
        };
    });

    const uint block_size = 64;
    auto test_shader = compile_async<1>(device, [&](BufferUInt queue, UInt queue_size,
                                                    BufferUInt queue_out1, BufferUInt queue_out1_size,
                                                    BufferUInt queue_out2, BufferUInt queue_out2_size,
                                                    Bool gen, UInt offset, UInt nxt1, UInt nxt2) noexcept {
        set_block_size(block_size, 1, 1);
        auto dispatch_id = dispatch_x();
        auto size = queue_size;
        $if(dispatch_id < size) {
            //$if(true){
            auto pixel_id = (dispatch_id) % pixel_count;
            auto sample_id = dispatch_id / pixel_count;
            auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
            Float u_test = 0.0f;
            UInt path_id = 0;
            $if(gen) {
                if (compact) {
                    path_id = offset + dispatch_id;
                } else {
                    path_id = queue.read(dispatch_id);
                }
                sampler()->start(pixel_coord, sample_id);
                auto u_filter = sampler()->generate_pixel_2d();
                auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
                u_test = sampler()->generate_1d();
                sampler()->save_state(path_id);
                auto camera_sample = camera->generate_ray(pixel_coord, 0.0f, u_filter, u_lens);
                path_states.write_ray(path_id, camera_sample.ray);
                light_samples.write_wi_and_pdf(path_id, make_float3(u_filter, u_filter[0]), u_test);
            }
            $else {
                path_id = queue.read(dispatch_id);
                sampler()->load_state(path_id);
                //auto ray = path_states.read_ray(path_id);
                //auto wi_and_pdf=light_samples.read_wi_and_pdf(path_id);
                //camera->film()->accumulate(pixel_coord, wi_and_pdf.xyz(), 1.f);
                u_test = sampler()->generate_1d();
                sampler()->save_state(path_id);
                //u_test = (dispatch_id % 11) * 0.1f;
            };
            auto condition = u_test < 0.9f;
            push_if(condition, path_id, queue_out1, queue_out1_size, gathering);
            push_if(!condition, path_id, queue_out2, queue_out2_size, gathering);
            if (gathering)
                path_states.write_kernel_index(path_id, ite(condition, nxt1, nxt2));
            /*    $if(condition) {
                    
                    auto queue_id = queue_out1_size.atomic(0u).fetch_add(1u);
                    //queue_out1.write(queue_id, path_id);
                }
                $else {
                    auto queue_id = queue_out2_size.atomic(0u).fetch_add(1u);
                    //queue_out2.write(queue_id, path_id);
                };*/



        };
    });
    // wait for the compilation of all shaders
    generate_rays_shader.get().set_name("generate_rays");
    intersect_shader.get().set_name("intersect");
    evaluate_miss_shader.get().set_name("evaluate_miss");
    evaluate_surface_shader.get().set_name("evaluate_surfaces");
    evaluate_light_shader.get().set_name("evaluate_lights");
    sample_light_shader.get().set_name("sample_lights");
    mark_invalid_shader.get().set_name("mark_invalid");
    gather_shader.get().set_name("gather");
    empty_gather_shader.get().set_name("empty_gather");
    compact_shader.get().set_name("compact");
    test_shader.get().set_name("test");
    ordering_shader.get().set_name("ordering");
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);

    LUISA_INFO("Rendering started.");
    // create path states

    auto shutter_samples = camera->node()->shutter_samples();

    Clock clock;
    ProgressBar progress_bar;
    progress_bar.update(0.0);
    auto launch_limit = state_count / (KERNEL_COUNT - 1);
    uint shutter_spp = 0;
    auto iteration = 0;
    auto gen_iter = 0;
    for (auto s : shutter_samples) {

        shutter_spp += s.spp;
        auto time = s.point.time;
        pipeline().update(command_buffer, time);
        aqueue.clear_counter_buffer(command_buffer);
        auto launch_state_count = s.spp * pixel_count;
        auto last_committed_state = launch_state_count;
        auto queues_empty = true;
        command_buffer << mark_invalid_shader.get()(aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID)).dispatch(state_count);

        //test case

        const uint test_iteration = 1193;
        auto local_iter = 0;
        if (test_case) {//test run
            LUISA_INFO("START TESTING...");
            for (auto it = 0u; it < test_iteration; ++it) {
                iteration++;
                local_iter++;
                //command_buffer << pipeline().printer().retrieve();
                //LUISA_INFO("get counters");

                aqueue.catch_counter(command_buffer);
                command_buffer << synchronize();//catch the queue counters
                auto max_count = 0u;
                auto max_index = -1;
                for (auto i = 0u; i < KERNEL_COUNT; ++i) {
                    //LUISA_INFO("queue {} counter:{}", i, queues[i].host_counter());
                    if (aqueue.host_counter(i) > 0) {

                        if (aqueue.host_counter(i) > max_count) {
                            max_count = aqueue.host_counter(i);
                            max_index = i;
                        }
                    }
                }
                LUISA_ASSERT(max_index != -1, "no path found error");
                auto test1 = max_index % KERNEL_COUNT;
                auto test2 = (max_index + 1) % KERNEL_COUNT;
                auto test3 = (max_index + 2) % KERNEL_COUNT;
                aqueue.clear_counter_buffer(command_buffer,test1);
                auto gen = (test1 == 0);
                if ((gen && local_iter >= 10) || it == 0) {
                    local_iter = 0;
                    gen = true;
                } else {
                    gen = false;
                }
                auto valid_count = state_count - aqueue.host_counter(0);

                if (gen) {
                    //LUISA_INFO("generating");
                    gen_iter++;
                    if (compact) {
                        empty_queue.clear_counter_buffer(command_buffer);
                        if (gathering) {
                            command_buffer << gather_shader.get()(aqueue.index_buffer(0),
                                                                  aqueue.counter_buffer(0),
                                                                  0, state_count)
                                                  .dispatch(luisa::align(state_count, gather_shader.get().block_size().x));
                            //queues[0].catch_counter(command_buffer);
                            //command_buffer << synchronize();
                            //LUISA_INFO("gathering get {} of kernel {}", queues[0].host_counter(), 0);
                            aqueue.clear_counter_buffer(command_buffer,0);
                        }
                        command_buffer << empty_gather_shader.get()(valid_count, aqueue.index_buffer(0), aqueue.host_counter(0),
                                                                    empty_queue.index_buffer(command_buffer), empty_queue.counter_buffer(command_buffer))
                                              .dispatch(aqueue.host_counter(0));//find all invalid with path_id<valid_count
                        empty_queue.clear_counter_buffer(command_buffer);
                        for (auto i = 1u; i < KERNEL_COUNT; ++i) {
                            if (aqueue.host_counter(i)) {
                                if (gathering) {
                                    aqueue.clear_counter_buffer(command_buffer,i);
                                    command_buffer << gather_shader.get()(aqueue.index_buffer(i),
                                                                          aqueue.counter_buffer(i),
                                                                          i, state_count)
                                                          .dispatch(luisa::align(state_count, gather_shader.get().block_size().x));
                                }
                                command_buffer << compact_shader.get()(valid_count, empty_queue.index_buffer(command_buffer), empty_queue.counter_buffer(command_buffer),
                                                                       aqueue.index_buffer(i), aqueue.counter_buffer(i))
                                                      .dispatch(aqueue.host_counter(i));//move every id>=valid_count to [0,valid_count-1]
                            }
                        }
                    }
                }
                LUISA_INFO("Launching test kernel {} with size {}", test1, aqueue.host_counter(test1));
                auto size = (aqueue.host_counter(test1) + block_size - 1) / block_size * block_size;
                if (gathering && !(gen & compact)) {
                    command_buffer << gather_shader.get()(aqueue.index_buffer(test1),
                                                          aqueue.counter_buffer(test1),
                                                          test1, state_count)
                                          .dispatch(luisa::align(state_count, gather_shader.get().block_size().x));
                    //queues[test1].catch_counter(command_buffer);
                    //command_buffer<< synchronize();
                    //LUISA_INFO("gathering get {} of kernel {}", queues[test1].host_counter(), test1);
                    aqueue.clear_counter_buffer(command_buffer,test1);
                }

                command_buffer << test_shader.get()(aqueue.index_buffer(test1), aqueue.host_counter(test1),
                                                    aqueue.index_buffer(test2), aqueue.counter_buffer(test2),
                                                    aqueue.index_buffer(test3), aqueue.counter_buffer(test3),
                                                    gen, valid_count, test2, test3)
                                      .dispatch(size);
            }
        } else {//actual rendering

            while (launch_state_count > 0 || !queues_empty) {
                iteration += 1;
                //command_buffer << pipeline().printer().retrieve();
                queues_empty = true;
                aqueue.catch_counter(command_buffer);

                for (auto i = 0u; i < KERNEL_COUNT; ++i) {
                    //LUISA_INFO("kernel {} has size {}", KernelName[i], aqueue.host_counter(i));
                }
                if (aqueue.host_counter(INVALID) > state_count / 2 && launch_state_count > 0) {//launch new kernel

                    auto generate_count = std::min(launch_state_count, aqueue.host_counter(INVALID));
                    auto zero = 0u;
                    //LUISA_INFO("Generate new kernel size {}", generate_count);
                    gen_iter += 1;
                    auto valid_count = state_count - aqueue.host_counter(INVALID);
                    if (gathering) {
                        aqueue.clear_counter_buffer(command_buffer,INVALID);
                        command_buffer << gather_shader.get()(aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID), INVALID, state_count)
                                                  .dispatch(luisa::align(state_count, gather_shader.get().block_size().x));
                    }
                    
                    aqueue.clear_counter_buffer(command_buffer,INVALID);
                    if (compact) {
                        empty_queue.clear_counter_buffer(command_buffer);
                        command_buffer << empty_gather_shader.get()(valid_count, aqueue.index_buffer(INVALID), aqueue.host_counter(INVALID),
                                                                    empty_queue.index_buffer(command_buffer), empty_queue.counter_buffer(command_buffer))
                                              .dispatch(aqueue.host_counter(INVALID));//find all invalid with path_id<valid_count
                        empty_queue.clear_counter_buffer(command_buffer);
                        for (auto i = 1u; i < KERNEL_COUNT; ++i) {
                            if (aqueue.host_counter(i)) {
                                if (gathering) {
                                    aqueue.clear_counter_buffer(command_buffer,i);
                                    command_buffer << gather_shader.get()(aqueue.index_buffer(i),
                                                                          aqueue.counter_buffer(i),
                                                                          i, state_count)
                                                          .dispatch(luisa::align(state_count, gather_shader.get().block_size().x));
                                }

                                command_buffer << compact_shader.get()(valid_count, empty_queue.index_buffer(command_buffer), empty_queue.counter_buffer(command_buffer),
                                                                       aqueue.index_buffer(i), aqueue.counter_buffer(i))
                                                      .dispatch(aqueue.host_counter(i));//move every id>=valid_count to [0,valid_count-1]
                            }
                        }
                        if (use_sort) {
                            auto offset = state_count;
                            for (auto i = 1u; i < KERNEL_COUNT; ++i) {
                                offset -= aqueue.host_counter(i);
                                if (aqueue.host_counter(i) != 0) {
                                    command_buffer << ordering_shader.get()(offset, aqueue.index_buffer(i),
                                                                            aqueue.host_counter(i))
                                                          .dispatch(aqueue.host_counter(i));
                                }
                            }
                        }
                    }
//                    command_buffer<<synchronize();
                    command_buffer << generate_rays_shader.get()(aqueue.index_buffer(INVALID), valid_count, aqueue.index_buffer(INTERSECT), aqueue.counter_buffer(INTERSECT),
                                                                 shutter_spp - s.spp, s.spp * pixel_count - launch_state_count,
                                                                 time, s.point.weight, generate_count)
                                          .dispatch(luisa::align(generate_count, generate_rays_shader.get().block_size().x));//generate rays in [valid_count,state_count)
                    launch_state_count -= generate_count;
                    queues_empty = false;
                    continue;
                }

                auto setup_workload = [&](uint max_index) {
                    if (gathering && !direct_launch) {
                        if (max_index == SURFACE && use_tag_sort) {
                            auto tag_size = pipeline().surfaces().size();
                            //LUISA_INFO("tag_size {}",tag_size);
                            command_buffer << bucket_update_shader.get()(light_samples.tag_counter(), tag_size).dispatch(1u);
                            command_buffer << sort_tag_gather_shader.get()(aqueue.index_buffer(max_index), light_samples.surface_tag(), light_samples.tag_counter(), max_index, tag_size).dispatch(state_count);
                            command_buffer << bucket_reset_shader.get()(light_samples.tag_counter()).dispatch(tag_size);
                        } else {//sorting kernel
                            aqueue.clear_counter_buffer(command_buffer,max_index);
                            command_buffer << gather_shader.get()(aqueue.index_buffer(max_index), aqueue.counter_buffer(max_index), max_index,state_count)
                                                  .dispatch(luisa::align(state_count, gather_shader.get().block_size().x));
                        }
                    }
                    aqueue.clear_counter_buffer(command_buffer,max_index);
                };
                auto launch_kernel = [&](uint max_index) {
                    auto dispatch_size = aqueue.host_counter(max_index);
                    if (direct_launch)
                        dispatch_size = state_count;
                    //LUISA_INFO("Launch kernel {} for size {}", KernelName[max_index], queues[max_index].host_counter());
                    switch (max_index) {
                        case INTERSECT:
                            command_buffer << intersect_shader.get()(aqueue.index_buffer(INTERSECT),
                                                                     aqueue.index_buffer(SAMPLE), aqueue.counter_buffer(SAMPLE),
                                                                     aqueue.index_buffer(LIGHT), aqueue.counter_buffer(LIGHT),
                                                                     aqueue.index_buffer(MISS), aqueue.counter_buffer(MISS),
                                                                     aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID))
                                                  .dispatch(dispatch_size);
                            break;
                        case MISS:
                            command_buffer << evaluate_miss_shader.get()(aqueue.index_buffer(MISS),
                                                                         aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID), time)
                                                  .dispatch(dispatch_size);
                            break;
                        case LIGHT:
                            command_buffer << evaluate_light_shader.get()(aqueue.index_buffer(LIGHT),
                                                                          aqueue.index_buffer(SAMPLE), aqueue.counter_buffer(SAMPLE),
                                                                          aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID), time)
                                                  .dispatch(dispatch_size);
                            break;
                        case SAMPLE:
                            command_buffer << sample_light_shader.get()(aqueue.index_buffer(SAMPLE),
                                                                        aqueue.index_buffer(SURFACE), aqueue.counter_buffer(SURFACE),
                                                                        aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID), time)
                                                  .dispatch(dispatch_size);
                            break;
                        case SURFACE:
                            command_buffer << evaluate_surface_shader.get()(aqueue.index_buffer(SURFACE),
                                                                            aqueue.index_buffer(INTERSECT), aqueue.counter_buffer(INTERSECT),
                                                                            aqueue.index_buffer(INVALID), aqueue.counter_buffer(INVALID), time)
                                                  .dispatch(dispatch_size);
                            break;
                        default:
                            LUISA_INFO("UNEXPECTED KERNEL INDEX");
                    }
                };

                auto max_count = 0u;
                auto max_index = -1;
                /*for (auto i = 1u; i < KERNEL_COUNT; ++i) {
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
                    setup_workload(max_index);
                    launch_kernel(max_index);
                    
                }*/
                for (auto i = 1u; i < KERNEL_COUNT; ++i) {
                    //LUISA_INFO("kernel queue {} has size {}", KernelName[i], queues[i].host_counter());
                    if (aqueue.host_counter(i) > 0) {
                        queues_empty = false;
                        setup_workload(i);
                    }
                }
                for (auto i = 1u; i < KERNEL_COUNT; ++i) {
                    //LUISA_INFO("kernel queue {} has size {}", KernelName[i], queues[i].host_counter());
                    if (aqueue.host_counter(i) > 0) {
                        launch_kernel(i);
                    }
                }
                auto launches_per_commit = 16u;
                if (last_committed_state - launch_state_count >= launches_per_commit * pixel_count) {
                    last_committed_state = launch_state_count;
                    auto p = (shutter_spp - last_committed_state / static_cast<double>(pixel_count)) / static_cast<double>(spp);
                    command_buffer << [p, &progress_bar] { progress_bar.update(p); };
                }
            }
        }
    }
    LUISA_INFO("Total iteration {}, where {} of them are generation", iteration, gen_iter);
    LUISA_INFO("Configuration: compact:{},gathering:{},\nuse_tag_sort:{},tot_surface_tag:{}\nuse_sort:{},direct_launch{}",
               compact, gathering, use_tag_sort, pipeline().surfaces().size(), use_sort, direct_launch);

    command_buffer << synchronize();
    progress_bar.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WavefrontPathTracingv2)
