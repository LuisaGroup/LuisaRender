//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "ray.h"
#include "device.h"
#include "film.h"
#include "node.h"
#include "parser.h"

namespace luisa {

class Camera : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Camera);

protected:
    std::shared_ptr<Film> _film;

public:
    explicit Camera(Device *device, const ParameterSet &parameters)
        : Node{device}, _film{parameters["film"].parse<Film>()} {}
    virtual void update(float time[[maybe_unused]]) { /* doing nothing by default */ }
    virtual void generate_rays(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, RayPool &ray_pool, RayQueueView ray_queue) = 0;
    [[nodiscard]] Film &film() { return *_film; }
};

}
