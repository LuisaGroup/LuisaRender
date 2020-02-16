//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/ray.h>

namespace luisa::path_tracing {



}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/integrator.h>

namespace luisa {

class PathTracing : public Integrator {

protected:
    std::unique_ptr<TypelessBuffer> _ray_state_buffer;
    std::unique_ptr<TypelessBuffer> _ray_pixel_buffer;
    std::unique_ptr<TypelessBuffer> _ray_sampler_state_buffer;
    std::unique_ptr<TypelessBuffer> _ray_throughput_buffer;
    std::unique_ptr<TypelessBuffer> _ray_radiance_buffer;
    std::unique_ptr<TypelessBuffer> _ray_depth_buffer;
    std::unique_ptr<TypelessBuffer> _ray_buffer;
    std::unique_ptr<TypelessBuffer> _shadow_ray_buffer;
    std::unique_ptr<TypelessBuffer> _closest_hit_buffer;
    std::unique_ptr<TypelessBuffer> _any_hit_buffer;
    std::unique_ptr<TypelessBuffer> _interaction_buffer;
    
    uint _max_depth;

public:
    PathTracing(Device *device, const ParameterSet &parameter_set);
    void render() override;
};

LUISA_REGISTER_NODE_CREATOR("PathTracing", PathTracing)

}

#endif
