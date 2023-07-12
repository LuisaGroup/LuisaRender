//
// Created by Mike Smith on 2022/1/10.
//

#include "compute/src/backends/common/hlsl/hlsl_codegen.h"
#include "dsl/builtin.h"
#include "dsl/struct.h"
#include "runtime/rtx/ray.h"
#include "util/spec.h"
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
class MegakernelWaveFront final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _block_count;
    bool _gathering;
    bool _test_case;
    bool _compact;
    bool _use_tag_sort;

public:
    MegakernelWaveFront(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _block_count{desc->property_uint_or_default("block_count", 4096u)},
          _gathering{desc->property_bool_or_default("gathering", true)},
          _use_tag_sort{desc->property_bool_or_default("use_tag_sort", true)},
          _test_case{desc->property_bool_or_default("test_case", false)},
          _compact{desc->property_bool_or_default("compact", true)} {}

    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto use_tag_sort() const noexcept { return _use_tag_sort; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto block_count() const noexcept { return _block_count; }
    [[nodiscard]] auto gathering() const noexcept { return _gathering; }
    [[nodiscard]] auto test_case() const noexcept { return _test_case; }
    [[nodiscard]] auto compact() const noexcept { return _compact; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
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
    AggregatedRayQueue(Device &device, size_t size, uint kernel_count, bool gathering) noexcept
        : _index_buffer{device.create_buffer<uint>(gathering ? size : kernel_count * size)},
          _counter_buffer{device.create_buffer<uint>(kernel_count)},
          _kernel_count{kernel_count},
          _size{size},
          _gathering{gathering} {
        _host_counter.resize(kernel_count);
        _offsets.resize(kernel_count);
        _clear_counters = device.compile<1>([this] {
            _counter_buffer->write(dispatch_x(), 0u);
        });
    }
    void clear_counter_buffer(CommandBuffer &command_buffer, int index = -1) noexcept {
        //if (_current_counter == counter_buffer_size-1) {
        //   _current_counter = 0u;
        if (index == -1) {
            command_buffer << _clear_counters().dispatch(_kernel_count);
        } else {
            static uint zero = 0u;
            command_buffer << counter_buffer(index).copy_from(&zero);
        }
        //} else
        //    _current_counter++;
    }
    [[nodiscard]] BufferView<uint> counter_buffer(uint index) noexcept {
        return _counter_buffer.view(index, 1);
    }
    [[nodiscard]] BufferView<uint> index_buffer(uint index) noexcept {
        if (_gathering)
            return _index_buffer.view(_offsets[index], _host_counter[index]);
        else
            return _index_buffer.view(index * _size, _size);
    }
    [[nodiscard]] uint host_counter(uint index) const noexcept {
        return _host_counter[index];
    }
    void catch_counter(CommandBuffer &command_buffer) noexcept {
        command_buffer << _counter_buffer.view(0, _kernel_count).copy_to(_host_counter.data());
        command_buffer << synchronize();
        uint prev = 0u;
        for (auto i = 0u; i < _kernel_count; ++i) {
            uint now = _host_counter[i];
            _offsets[i] = prev;
            prev += now;
        }
    }
};
struct ThreadFrame {
    float wl_sample;
    float pdf_bsdf;
    uint kernel_index;
    uint depth;
    uint pixel_index;
    float4 wi_and_pdf;
};
struct DimensionalFrame {
    float beta;
    float emission;
};
}// namespace luisa::render
LUISA_STRUCT(luisa::render::ThreadFrame, wl_sample, pdf_bsdf, kernel_index, depth, pixel_index, wi_and_pdf){};
LUISA_STRUCT(luisa::render::DimensionalFrame, beta, emission){};
namespace luisa::render {
class StateSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<ThreadFrame> _frame;
    Buffer<DimensionalFrame> _dim_frame;
    Buffer<Ray> _ray;
    Buffer<Hit> _hit;

public:
    StateSOA(const Spectrum::Instance *spectrum,size_t size) noexcept
        : _spectrum{spectrum} {
        auto &&device = spectrum->pipeline().device();
        auto dimension = spectrum->node()->dimension();
        _dim_frame = device.create_buffer<DimensionalFrame>(size * dimension);
        _frame = device.create_buffer<ThreadFrame>(size);
        _ray = device.create_buffer<Ray>(size);
        _hit = device.create_buffer<Hit>(size);
    }
    [[nodiscard]] auto read_ray(Expr<uint> index) const noexcept {
        return _ray->read(index);
    }
    [[nodiscard]] auto read_hit(Expr<uint> index) const noexcept {
        return _hit->read(index);
    }
    [[nodiscrad]] auto read_frame(Expr<uint> index) const noexcept {
		return _frame->read(index);
	}
    [[nodiscard]] auto read_dim_frame(Expr<uint> index) const noexcept {
		return _dim_frame->read(index);
	}

    void write_ray(Expr<uint> index, Expr<Ray> ray) noexcept {
        _ray->write(index, ray);
    }
    void write_hit(Expr<uint> index, Expr<Hit> hit) noexcept {
        _hit->write(index, hit);
    }
    void write_frame(Expr<uint> index, Expr<ThreadFrame> frame) noexcept {
		_frame->write(index, frame);
	}
    void write_dim_frame(Expr<uint> index, Expr<DimensionalFrame> frame) noexcept {
        _dim_frame->write(index, frame);
    }
    


#define MOVE(entry, from, to)       \
  {                                 \
    auto inst = read_##entry(from); \
    write_##entry(to, inst);        \
  }
#undef MOVE
};
class MegakernelWaveFrontInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override;
};

luisa::unique_ptr<Integrator::Instance> MegakernelWaveFront::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelWaveFrontInstance>(pipeline, command_buffer, this);
}

void MegakernelWaveFrontInstance::_render_one_camera(
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
    auto gathering = node<MegakernelWaveFront>()->gathering();
    auto test_case = node<MegakernelWaveFront>()->test_case();
    auto compact = node<MegakernelWaveFront>()->compact();
    auto use_tag_sort = node<MegakernelWaveFront>()->use_tag_sort();
    bool use_sort = true;
    bool direct_launch = false;

    auto spectrum = pipeline().spectrum();

    Clock clock_compile;
    //assume KERNEL_COUNT< block_size_x
    auto block_size = 128u;
    auto sm_count = 128u;
    auto launch_size = sm_count*block_size;
    auto q_factor = 1u;
    auto g_factor = KERNEL_COUNT - q_factor;
    StateSOA state_soa{spectrum, launch_size*g_factor};
    sampler()->reset(command_buffer, resolution, (KERNEL_COUNT+1)*launch_size, spp);
    command_buffer << synchronize();
    auto render_shader = compile_async<1>(device, [&](BufferUInt samples, UInt tot_samples, UInt base_spp, Float time, Float shutter_weight) noexcept {
        set_block_size(block_size, 1, 1);
        const uint fetch_size = 128;
        auto dim = spectrum->node()->dimension();
        auto shared_queue_size = block_size * q_factor;
        auto queue_size = block_size * (KERNEL_COUNT+1);
        Shared<ThreadFrame> path_state{shared_queue_size};
        Shared<Ray> path_ray{shared_queue_size};
        Shared<Hit> path_hit{shared_queue_size};
        Shared<DimensionalFrame> path_state_dim{shared_queue_size * dim};
        Shared<uint> path_id{shared_queue_size};
        Shared<uint> work_counter{KERNEL_COUNT};
        Shared<uint> work_offset{2u};
        Shared<uint> workload{2};
        Shared<uint> work_stat{2};//0 max_count,1 max_id
        path_state[thread_x()].kernel_index = (uint)INVALID;
        $if(thread_x() < (uint)KERNEL_COUNT) {
            $if(thread_x() == 0){
                work_counter[thread_x()] = KERNEL_COUNT*block_size;
            }
            $else {
                work_counter[thread_x()] = 0u;
            };
        };
        workload[0] = 0;
        workload[1] = 0;
        //Shared<uint> count{1u};
        auto count = def(0);
        Shared<uint> rem_global{1};
        Shared<uint> rem_local{1};
        rem_global[0] = 1u;
        rem_local[0] = 0u;
        sync_block();
        //pipeline().printer().info("work counter {} of block {}: {}", -1, block_x(), -1);
        auto count_limit = 1000000u;
        
        $while((rem_global[0] != 0u | rem_local[0] != 0u) & (count!= count_limit)) {
            sync_block();//very important, synchronize for condition
            rem_local[0] = 0u;
            count += 1;
            work_stat[0] = 0;
            work_stat[1] = -1;
            /* $if(thread_x() < (uint)KERNEL_COUNT) {//clear counter
                work_counter[thread_x()] = 0u;
            };
            sync_block();
            $for(index, 0u, q_factor) {
                for (auto i = 0u; i < KERNEL_COUNT; ++i) {//count the kernels
                    auto state = path_state[index*block_size+thread_x()];
                    $if(state.kernel_index == i) {
                        if (i != (uint)INVALID) {
                            rem_local[0] = 1u;
                        } else {
                            $if(workload[0] < workload[1]) {
                                rem_local[0] = 1u;
                            };
                        }
                        work_counter.atomic(i).fetch_add(1u);
                    };
                }
            };*/
            sync_block();
            $if(thread_x() == block_size - 1) {
                $if((workload[0] >= workload[1]) & (rem_global[0]==1u)) {//fetch new workload
                    workload[0] = samples.atomic(0u).fetch_add(block_size * fetch_size);
                    workload[1] = min(workload[0] + block_size * fetch_size, tot_samples);
                    $if(workload[0] >= tot_samples) {
                        rem_global[0] = 0u;
                    };
                };
            };
            sync_block();
            $if(thread_x() < (uint)KERNEL_COUNT) {//get max
                $if((workload[0] < workload[1]) | (thread_x() != 0u)) {
                    $if(work_counter[thread_x()] != 0) {
                        rem_local[0] = 1u;
                        work_stat.atomic(0).fetch_max(work_counter[thread_x()]);
                    };
                };
                //pipeline().printer().info("work counter {} of block {}: {}", thread_x(), block_x(), work_counter[thread_x()]);

			};
            sync_block();
            $if(thread_x() < (uint)KERNEL_COUNT){//get argmax
                $if((work_stat[0] == work_counter[thread_x()]) & ((workload[0] < workload[1]) | (thread_x() != 0u))) {
                    work_stat[1] = thread_x();
                };
            };
            sync_block();
            work_offset[0] = 0;
            work_offset[1] = 0;
            sync_block();
            $for(index, 0u, q_factor) {//collect indices
                auto state = path_state[index * block_size + thread_x()];
                $if(state.kernel_index != work_stat[1]) {
                    auto id=work_offset.atomic(0).fetch_add(1u);
                    path_id[id]=index * block_size + thread_x();
                };
            };
            sync_block();
            $if(q_factor * block_size - work_offset[0] < block_size) {//no enough work
                $for(index, 0u, g_factor) {//swap frames
                    auto soa_id = block_x() * (block_size * g_factor) + index * block_size + thread_x();
                    auto state = state_soa.read_frame(soa_id);
                    $if(state.kernel_index == work_stat[1]) {
                        auto id = work_offset.atomic(1).fetch_add(1u);
                        $if(id < work_offset[0]) {
                            auto dst = path_id[id];
                            $if(state.kernel_index != (uint)INVALID) {
                                $if(path_state[dst].kernel_index != (uint)INVALID) {
                                    auto ray = state_soa.read_ray(soa_id);
                                    state_soa.write_ray(soa_id, path_ray[dst]);
                                    path_ray[dst] = ray;
                                    auto hit = state_soa.read_hit(soa_id);
                                    state_soa.write_hit(soa_id, path_hit[dst]);
                                    path_hit[dst] = hit;
                                    for (auto i = 0u; i < dim; ++i) {
                                        auto dim_frame = state_soa.read_dim_frame(soa_id * dim + i);
                                        state_soa.write_dim_frame(soa_id * dim + i, path_state_dim[dst * dim + i]);
                                        path_state_dim[dst * dim + i] = dim_frame;
                                    }
                                    sampler()->load_state(queue_size * block_x() + block_size*q_factor+index*block_size+thread_x());
                                    sampler()->save_state(queue_size * block_x() + KERNEL_COUNT*block_size+thread_x());
                                    sampler()->load_state(queue_size * block_x() + dst);
                                    sampler()->save_state(queue_size * block_x() + block_size*q_factor+index*block_size+thread_x());
                                    sampler()->load_state(queue_size * block_x() + KERNEL_COUNT*block_size+thread_x());
                                    sampler()->save_state(queue_size * block_x() + dst);
                                }
                                $else {
                                    auto ray = state_soa.read_ray(soa_id);
									path_ray[dst] = ray;
									auto hit = state_soa.read_hit(soa_id);
									path_hit[dst] = hit;
                                    for (auto i = 0u; i < dim; ++i) {
										auto dim_frame = state_soa.read_dim_frame(soa_id * dim + i);
										path_state_dim[dst * dim + i] = dim_frame;
									}
                                    sampler()->load_state(queue_size * block_x() + block_size * q_factor + index * block_size + thread_x());
                                    sampler()->save_state(queue_size * block_x() + dst);
								};
                            }
                            $else {
                                $if(path_state[dst].kernel_index != (uint)INVALID){
                                    state_soa.write_ray(soa_id, path_ray[dst]);
                                    state_soa.write_hit(soa_id, path_hit[dst]);
                                    for (auto i = 0u; i < dim; ++i) {
                                        state_soa.write_dim_frame(soa_id * dim + i, path_state_dim[dst * dim + i]);
                                    }
                                    sampler()->load_state(queue_size * block_x() + dst);
                                    sampler()->save_state(queue_size * block_x() + block_size * q_factor + index * block_size + thread_x());     
                                };
                            };
                            state_soa.write_frame(soa_id, path_state[dst]);
                            path_state[dst] = state;
                            
                            
                        };
                    };
                };
            };
            auto save_kernel = [&](UInt path_id, KernelState src, KernelState dst) noexcept {
                path_state[path_id].kernel_index = (uint)dst;
                work_counter.atomic((uint)src).fetch_sub(1u);
                work_counter.atomic((uint)dst).fetch_add(1u);
			};
            auto generate_ray_shader = [&](UInt path_id, UInt work_id) noexcept {//TODO: add fetch_state and set_state for sampler
                auto pixel_id = work_id % pixel_count;
                auto sample_id = base_spp + work_id / pixel_count;
                auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                camera->film()->accumulate(pixel_coord, make_float3(0.f), 1.f);
                sampler()->start(pixel_coord, sample_id);
                auto u_filter = sampler()->generate_pixel_2d();
                auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
                auto u_wavelength = spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d();
                sampler()->save_state(queue_size * block_x() + path_id);
                auto camera_sample = camera->generate_ray(pixel_coord, time, u_filter, u_lens);
                path_ray[path_id] = camera_sample.ray;
                path_state[path_id].wl_sample = u_wavelength;
                $for(i, 0u, dim) {
                    path_state_dim[path_id * dim + i].beta = camera_sample.weight * shutter_weight;
                };
                path_state[path_id].pdf_bsdf = 1e16f;
                path_state[path_id].pixel_index = pixel_id;
                path_state[path_id].depth = 0u;
                //pipeline().printer().info("path id:{}, pixel:{}", path_id, work_id);
                save_kernel(path_id, INVALID, INTERSECT);
                workload.atomic(0).fetch_add(1u);
            };

            auto intersect_shader = [&](UInt path_id) noexcept {
                auto ray = path_ray[path_id];
                auto hit = pipeline().geometry()->trace_closest(ray);
                path_hit[path_id] = hit;
                $if(!hit->miss()) {
                    auto shape = pipeline().geometry()->instance(hit.inst);
                    $if(shape.has_light()) {
                        save_kernel(path_id, INTERSECT,LIGHT);
                    }
                    $else {
                        $if(shape.has_surface()) {
                            save_kernel(path_id, INTERSECT, SAMPLE);

                        }
                        $else {
                            save_kernel(path_id, INTERSECT, INVALID);

                        };
                    };
                }
                $else {
                    if (pipeline().environment()) {
                        save_kernel(path_id, INTERSECT, MISS);


                    } else {
                        save_kernel(path_id, INTERSECT, INVALID);
                    }
                };
            };
            auto test_shader = [&](UInt path_id) noexcept {
                sampler()->load_state(queue_size * block_x() + path_id);
                auto u_test = sampler()->generate_1d();
                sampler()->save_state(queue_size * block_x() + path_id);
                $if (u_test < 0.2f) {
                    save_kernel(path_id, INTERSECT, INVALID);
                } $else{
					save_kernel(path_id, INTERSECT, INTERSECT);
				};
            };
                
            auto evaluate_miss_shader = [&](UInt path_id) noexcept {
                if (pipeline().environment()) {
                    auto wi = path_ray[path_id]->direction();
                    auto u_wl = def(0.f);
                    if (!spectrum->node()->is_fixed()) {
                        u_wl = path_state[path_id].wl_sample;
                    }
                    auto swl = spectrum->sample(abs(u_wl));
                    $if(u_wl < 0.f) { swl.terminate_secondary(); };
                    auto pdf_bsdf = path_state[path_id].pdf_bsdf;
                    SampledSpectrum beta{dim};
                    for (auto i = 0u; i < dim; ++i) {
                        beta[i] = path_state_dim[path_id * dim + i].beta;
                    }
                    auto eval = light_sampler()->evaluate_miss(wi, swl, time);
                    auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                    auto Li = beta * eval.L * mis_weight;
                    auto pixel_id = path_state[path_id].pixel_index;
                    auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                    camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
                }
                save_kernel(path_id, MISS, INVALID);

            };

            auto evaluate_light_shader = [&](UInt path_id) noexcept {
                if (!pipeline().lights().empty()) {
                    auto ray = path_ray[path_id];
                    auto hit = path_hit[path_id];
                    auto u_wl = def(0.f);
                    if (!spectrum->node()->is_fixed()) {
                        u_wl = path_state[path_id].wl_sample;
                    }
                    auto swl = spectrum->sample(abs(u_wl));
                    $if(u_wl < 0.f) { swl.terminate_secondary(); };
                    auto pdf_bsdf = path_state[path_id].pdf_bsdf;
                    SampledSpectrum beta{dim};
                    for (auto i = 0u; i < dim; ++i) {
                        beta[i] = path_state_dim[path_id * dim + i].beta;
                    }
                    auto it = pipeline().geometry()->interaction(ray, hit);
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                    auto Li = beta * eval.L * mis_weight;
                    auto pixel_id = path_state[path_id].pixel_index;
                    auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                    camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
                    auto shape = pipeline().geometry()->instance(hit.inst);
                    $if(shape.has_surface()) {
                        save_kernel(path_id, LIGHT, SAMPLE);

                    }
                    $else {
                        save_kernel(path_id, LIGHT, INVALID);

                    };
                } else {
                    save_kernel(path_id, LIGHT, INVALID);

                }
            };

            auto sample_light_shader = [&](UInt path_id) noexcept {
                sampler()->load_state(queue_size * block_x() + path_id);
                auto u_light_selection = sampler()->generate_1d();
                auto u_light_surface = sampler()->generate_2d();
                sampler()->save_state(queue_size * block_x() + path_id);
                auto ray = path_ray[path_id];
                auto hit = path_hit[path_id];
                auto it = pipeline().geometry()->interaction(ray, hit);
                auto u_wl = def(0.f);
                if (!spectrum->node()->is_fixed()) {
                    u_wl = path_state[path_id].wl_sample;
                }
                auto swl = spectrum->sample(abs(u_wl));
                $if(u_wl < 0.f) { swl.terminate_secondary(); };
                auto light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);
                // trace shadow ray
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);//if occluded, transit to invalid
                $for(i, 0u, dim) {
                    path_state_dim[path_id * dim + i].emission = ite(occluded, 0.f, 1.f) * light_sample.eval.L[i];
                };
                path_state[path_id].wi_and_pdf = make_float4(light_sample.shadow_ray->direction(),
                                                             ite(occluded, 0.f, light_sample.eval.pdf));
                save_kernel(path_id, SAMPLE, SURFACE);

            };

            auto evaluate_surface_shader = [&](UInt path_id) noexcept {
                sampler()->load_state(queue_size * block_x() + path_id);
                auto depth = path_state[path_id].depth;
                auto u_lobe = sampler()->generate_1d();
                auto u_bsdf = sampler()->generate_2d();
                auto u_rr = def(0.f);
                auto rr_depth = node<MegakernelWaveFront>()->rr_depth();
                $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
                sampler()->save_state(queue_size * block_x() + path_id);

                auto ray = path_ray[path_id];
                auto hit = path_hit[path_id];
                auto it = pipeline().geometry()->interaction(ray, hit);
                auto u_wl = def(0.f);
                if (!spectrum->node()->is_fixed()) {
                    u_wl = path_state[path_id].wl_sample;
                }
                auto swl = spectrum->sample(abs(u_wl));
                $if(u_wl < 0.f) { swl.terminate_secondary(); };
                SampledSpectrum beta{dim};
                for (auto i = 0u; i < dim; ++i) {
                    beta[i] = path_state_dim[path_id * dim + i].beta;
                }
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
                        path_state[path_id].pdf_bsdf = 1e16f;
                    }
                    $else {
                        if (auto dispersive = closure->is_dispersive()) {
                            $if(*dispersive) {
                                swl.terminate_secondary();
                                if (!spectrum->node()->is_fixed()) {
                                    path_state[path_id].wl_sample = -u_wl;
                                }
                            };
                        }
                        // direct lighting
                        auto light_wi_and_pdf = path_state[path_id].wi_and_pdf;
                        auto pdf_light = light_wi_and_pdf.w;
                        $if(light_wi_and_pdf.w > 0.f) {
                            auto eval = closure->evaluate(wo, light_wi_and_pdf.xyz());
                            auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                            // update Li
                            SampledSpectrum Ld{dim};
                            for (auto i = 0u; i < dim; ++i) {
                                Ld[i] = path_state_dim[path_id * dim + i].emission;
                            }
                            auto Li = mis_weight / pdf_light * beta * eval.f * Ld;
                            auto pixel_id = path_state[path_id].pixel_index;
                            auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                            camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
                        };
                        // sample material
                        auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                        path_state[path_id].pdf_bsdf = surface_sample.eval.pdf;
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
                    auto rr_threshold = node<MegakernelWaveFront>()->rr_threshold();
                    auto q = max(beta.max() * eta_scale, 0.05f);
                    $if(depth + 1u >= rr_depth) {
                        terminated = q < rr_threshold & u_rr >= q;
                        beta *= ite(q < rr_threshold, 1.f / q, 1.f);
                    };
                };
                $if(depth + 1 >= node<MegakernelWaveFront>()->max_depth()) {
                    terminated = true;
                };
                auto pixel_id = path_state[path_id].pixel_index;
                auto pixel_coord = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                Float termi = 0.f;
                $if(terminated) {
                    termi = 1.f;
                };

                $if(!terminated) {
                    path_state[path_id].depth = depth + 1;
                    for (auto i = 0u; i < dim; ++i) {
                        path_state_dim[path_id * dim + i].beta = beta[i];
                    }
                    path_ray[path_id] = ray;
                    save_kernel(path_id, SURFACE, INTERSECT);

                }
                $else {
                    save_kernel(path_id, SURFACE, INVALID);
                };
            };
            auto gen_st = workload[0];
            sync_block();
            auto pid = thread_x();
            //pipeline().printer().info("loop {},block {} thread{},genwork {}~{}, processing pid {}, kernel {}",
            //        count, block_x(), thread_x(), workload[0], workload[1], pid, path_state[pid].kernel_index);
            $if(true) {
                $switch(path_state[pid].kernel_index) {
                    $case((uint)INVALID) {
                        $if(gen_st + thread_x() < workload[1]) {
                            generate_ray_shader(pid, gen_st + thread_x());//only work when kernel 0s are continue
                            //pipeline().printer().info("load {}, counter {},work_id {}", workload[0],gen_count,workload[0] + thread_x());
                        };
                    };
                    $case((uint)INTERSECT) {
                        intersect_shader(pid);
                    };
                    $case((uint)MISS) {
                        evaluate_miss_shader(pid);
                    };
                    $case((uint)LIGHT) {
                        evaluate_light_shader(pid);
                    };
                    $case((uint)SAMPLE) {
                        sample_light_shader(pid);
                    };
                    $case((uint)SURFACE) {
                        evaluate_surface_shader(pid);
                    };
                };
            };
            sync_block();
        };
        $if(count == count_limit) {
            pipeline().printer().info("block_id{},thread_id {}, loop not break! local:{}, global:{}",block_x(),thread_x(), rem_local[0], rem_global[0]);
            $if(thread_x() < (uint)KERNEL_COUNT){
                pipeline().printer().info("work rem: id {}, size {}", thread_x(), work_counter[thread_x()]);
            };
		};
    });
    auto clear_global_shader = compile_async<1>(device, [&]() noexcept {
        auto dispatch_id = dispatch_x();
        auto frame= state_soa.read_frame(0u);
        frame.kernel_index = (uint)INVALID;
        state_soa.write_frame(dispatch_id,frame);
    });
    render_shader.get().set_name("render");
    clear_global_shader.get().set_name("clear_global");
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);

    LUISA_INFO("Rendering started.");
    // create path states

    auto shutter_samples = camera->node()->shutter_samples();

    Clock clock;
    ProgressBar progress_bar;
    progress_bar.update(0.0);
    uint shutter_spp = 0;
    auto sample_count = device.create_buffer<uint>(1u);

    command_buffer << synchronize();

    for (auto s : shutter_samples) {
        uint host_sample_count = s.spp * pixel_count;
        static uint zero = 0u;
        auto time = s.point.time;
        pipeline().update(command_buffer, time);
        command_buffer << clear_global_shader.get()().dispatch(launch_size * g_factor);
        command_buffer << sample_count.copy_from(&zero)
                       << commit();
        LUISA_ASSERT(launch_size % render_shader.get().block_size().x == 0u, "");
        command_buffer << render_shader.get()(sample_count, host_sample_count, shutter_spp, time, s.point.weight).dispatch(launch_size);
        command_buffer << pipeline().printer().retrieve();
        command_buffer << synchronize();
        shutter_spp += s.spp;
    }

    progress_bar.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelWaveFront)
