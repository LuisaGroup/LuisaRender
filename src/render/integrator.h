//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "ray.h"
#include <compute/mathematics.h>
#include "viewport.h"
#include "core/parser.h"
#include "scene.h"
#include "camera.h"
#include "sampler.h"

namespace luisa {

class Integrator : public Plugin {

protected:
    Scene *_scene{nullptr};
    Camera *_camera{nullptr};
    Sampler *_sampler{nullptr};
    Viewport _viewport{};
    
    virtual void _prepare_for_frame() = 0;

public:
    Integrator(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) noexcept
        : Plugin{device} {}
    
    void prepare_for_frame(Scene *scene, Camera *camera, Sampler *sampler, Viewport viewport) {
        _scene = scene;
        _camera = camera;
        _sampler = sampler;
        _viewport = viewport;
        _prepare_for_frame();
    }
    
    virtual void render_frame(KernelDispatcher &dispatch) = 0;
};

}
