//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/ray.h>

namespace luisa::integrator::path_tracing {



}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/integrator.h>
#include <core/hit.h>

namespace luisa {

class PathTracing : public Integrator {

protected:
    std::unique_ptr<Buffer<float2>> _ray_pixel_buffer;
    std::unique_ptr<Buffer<float3>> _ray_throughput_buffer;
    std::unique_ptr<Buffer<float3>> _ray_radiance_buffer;
    std::unique_ptr<Buffer<uint8_t>> _ray_depth_buffer;
    std::unique_ptr<Buffer<float>> _ray_pdf_buffer;
    std::unique_ptr<Buffer<Ray>> _ray_buffer;
    
    uint _max_depth;

public:
    PathTracing(Device *device, const ParameterSet &parameter_set);
    void render_frame(Viewport viewport, Scene &scene, Camera &camera, Sampler &sampler) override;
};

LUISA_REGISTER_NODE_CREATOR("PathTracing", PathTracing)

}

#endif
