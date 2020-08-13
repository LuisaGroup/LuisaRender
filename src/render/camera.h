//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <core/data_types.h>
#include "viewport.h"

namespace luisa::camera {

struct GeneratePixelSamplesWithoutFilterKernelUniforms {
    Viewport tile_viewport;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "ray.h"
#include <compute/device.h>
#include "film.h"
#include "plugin.h"
#include "parser.h"
#include "sampler.h"
#include "transform.h"

namespace luisa {

class Camera : public Plugin {

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
        : Plugin{device},
          _film{parameters["film"].parse<Film>()},
          _transform{parameters["transform"].parse_or_null<Transform>()},
          _generate_pixel_samples_without_filter_kernel{device->load_kernel("camera::generate_pixel_samples_without_filter")} {}
    
    virtual void update(float time) {
        if (_transform != nullptr) { _camera_to_world = _transform->dynamic_matrix(time) * _transform->static_matrix(); }
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
