//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "ray.h"
#include "device.h"
#include "film.h"
#include "node.h"

namespace luisa {

class Camera : public Node {

protected:
    std::unique_ptr<Film> _film;

public:
    explicit Camera(Device *device, std::unique_ptr<Film> film) : Node{device}, _film{std::move(film)} {}
    virtual void update(float time) = 0;
    virtual void generate_rays(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue) = 0;
    [[nodiscard]] Film &film() { return *_film; }
};

}
