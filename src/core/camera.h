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

namespace luisa {

class Camera : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Camera);

protected:
    std::shared_ptr<Film> _film;

public:
    Camera(Device *device, const ParameterSet &parameters)
        : Node{device}, _film{parameters["film"].parse<Film>()} {}
    virtual void update(float time[[maybe_unused]]) { /* doing nothing by default */ }
    
    virtual void generate_rays(KernelDispatcher &dispatch,
                               Sampler &sampler,
                               BufferView<uint> ray_queue,
                               BufferView<uint> ray_queue_size,
                               BufferView<float2> pixel_buffer,
                               BufferView<Ray> ray_buffer,
                               BufferView<float3> throughput_buffer) = 0;
    
    [[nodiscard]] Film &film() { return *_film; }

};

}
