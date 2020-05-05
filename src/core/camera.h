//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "viewport.h"

namespace luisa::camera {

struct GeneratePixelSamplesWithoutFilterKernelUniforms {
    Viewport tile_viewport;
};

LUISA_DEVICE_CALLABLE inline void generate_pixel_samples_without_filter(
    LUISA_DEVICE_SPACE const float2 *sample_buffer,
    LUISA_DEVICE_SPACE float2 *pixel_buffer,
    LUISA_DEVICE_SPACE float3 *throughput_buffer,
    LUISA_UNIFORM_SPACE GeneratePixelSamplesWithoutFilterKernelUniforms &uniforms,
    uint tid) {
    
    if (tid < uniforms.tile_viewport.size.x * uniforms.tile_viewport.size.y) {
        auto offset_x = static_cast<float>(tid % uniforms.tile_viewport.size.x);
        auto offset_y = static_cast<float>(tid / uniforms.tile_viewport.size.x);
        pixel_buffer[tid] = make_float2(uniforms.tile_viewport.origin + make_uint2(offset_x, offset_y)) + sample_buffer[tid];
        throughput_buffer[tid] = make_float3(1.0f);
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "ray.h"
#include "device.h"
#include "film.h"
#include "node.h"
#include "parser.h"
#include "sampler.h"
#include "transform.h"

namespace luisa {

class Camera : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Camera);

protected:
    std::shared_ptr<Film> _film;
    std::shared_ptr<Transform> _transform;
    float4x4 _camera_to_world{1.0f};
    std::unique_ptr<Kernel> _generate_pixel_samples_without_filter_kernel;
    
    virtual void _generate_rays(KernelDispatcher &dispatch,
                                Sampler &sampler,
                                Viewport tile_viewport,
                                BufferView<float2> pixel_buffer,
                                BufferView<Ray> ray_buffer,
                                BufferView<float3> throughput_buffer) = 0;

public:
    Camera(Device *device, const ParameterSet &parameters)
        : Node{device},
          _film{parameters["film"].parse<Film>()},
          _transform{parameters["transform"].parse_or_null<Transform>()},
          _generate_pixel_samples_without_filter_kernel{device->load_kernel("camera::generate_pixel_samples_without_filter")} {}
    
    virtual void update(float time) {
        if (_transform != nullptr) {
            _camera_to_world = _transform->dynamic_matrix(time) * _transform->static_matrix();
        }
    }
    
    virtual void generate_rays(KernelDispatcher &dispatch,
                               Sampler &sampler,
                               Viewport tile_viewport,
                               BufferView<float2> pixel_buffer,
                               BufferView<Ray> ray_buffer,
                               BufferView<float3> throughput_buffer);
    
    [[nodiscard]] Film &film() { return *_film; }
    
};

}

#endif
