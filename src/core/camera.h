//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

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
    float4x4 _camera_to_world{};

public:
    Camera(Device *device, const ParameterSet &parameters)
        : Node{device},
        _film{parameters["film"].parse<Film>()},
        _transform{parameters["transform"].parse_or_null<Transform>()} {
        
        if (_transform == nullptr) {
            _transform = std::make_shared<Transform>(_device);
        }
    }
    
    virtual void update(float time) { _camera_to_world = _transform->dynamic_matrix(time) * _transform->static_matrix(); }
    virtual void generate_rays(KernelDispatcher &dispatch,
                               Sampler &sampler,
                               Viewport tile_viewport,
                               BufferView<float2> pixel_buffer,
                               BufferView<Ray> ray_buffer,
                               BufferView<float3> throughput_buffer) = 0;
    
    [[nodiscard]] Film &film() { return *_film; }

};

}
