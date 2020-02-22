//
// Created by Mike Smith on 2020/2/21.
//

#pragma once

#include "data_types.h"
#include "mathematics.h"
#include "light.h"

namespace luisa::illumination {

class Info {

private:
    uint8_t _tag{};
    uint8_t _index_hi{};
    uint16_t _index_lo{};

#ifndef LUISA_DEVICE_COMPATIBLE
public:
    constexpr Info(uint tag, uint index) noexcept
        : _tag{static_cast<uint8_t>(tag)}, _index_hi{static_cast<uint8_t>(index >> 24u)}, _index_lo{static_cast<uint16_t>(index)} {}
#endif

public:
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto tag() const noexcept { return static_cast<uint>(_tag); }
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto index() const noexcept { return (static_cast<uint>(_index_hi) << 24u) | static_cast<uint>(_index_lo); }
};

struct SelectLightsKernelUniforms {
    uint light_count;
    uint max_queue_size;
};

LUISA_DEVICE_CALLABLE inline void uniform_select_lights(
    LUISA_DEVICE_SPACE const float *sample_buffer,
    LUISA_DEVICE_SPACE const Info *info_buffer,
    LUISA_DEVICE_SPACE Atomic<uint> *queue_sizes,
    LUISA_DEVICE_SPACE light::Selection *queues,
    uint ray_count,
    SelectLightsKernelUniforms uniforms,
    uint tid) {
    
    if (tid < ray_count) {
        auto light_info = info_buffer[min(static_cast<uint>(sample_buffer[tid] * uniforms.light_count), uniforms.light_count - 1u)];
        auto queue_index = luisa_atomic_fetch_add(queue_sizes[light_info.tag()], 1u);
        queues[light_info.tag() * uniforms.max_queue_size + queue_index] = {light_info.index(), tid};
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <vector>

#include "geometry.h"
#include "kernel.h"
#include "viewport.h"

namespace luisa {

class Illumination {

private:
    Device *_device;
    Geometry *_geometry;
    std::vector<std::shared_ptr<Light>> _lights;
    
    std::unique_ptr<Buffer<illumination::Info>> _info_buffer;
    std::vector<uint> _light_sampling_dimensions;
    std::vector<std::unique_ptr<Kernel>> _light_sampling_kernels;
    std::vector<Light::SampleLightsDispatch> _light_sampling_dispatches;
    std::vector<std::unique_ptr<TypelessBuffer>> _light_data_buffers;
    
    // kernels
    std::unique_ptr<Kernel> _uniform_select_lights_kernel;

public:
    Illumination(Device *device, std::vector<std::shared_ptr<Light>> lights, Geometry *geometry);
    [[nodiscard]] uint tag_count() const noexcept { return static_cast<uint>(_light_data_buffers.size()); }
    
    void uniform_select_lights(KernelDispatcher &dispatch,
                               uint max_ray_count,
                               BufferView<uint> ray_queue,
                               BufferView<uint> ray_queue_size,
                               Sampler &sampler,
                               BufferView<light::Selection> queues,
                               BufferView<uint> queue_sizes);
    
    void sample_lights(KernelDispatcher &dispatch,
                       Sampler &sampler,
                       BufferView<uint> ray_indices,
                       BufferView<uint> ray_count,
                       BufferView<light::Selection> queues,
                       BufferView<uint> queue_sizes,
                       uint max_queue_size,
                       InteractionBufferSet &interactions,
                       LightSampleBufferSet &light_samples);
};

}

#endif
