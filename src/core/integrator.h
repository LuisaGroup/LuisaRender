//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "ray.h"
#include "mathematics.h"

namespace luisa::integrator {

struct PrepareForTileUniforms {
    uint2 resolution;
    uint4 viewport;  // (x, y, w, h)
};

LUISA_DEVICE_CALLABLE inline void prepare_for_tile(
    LUISA_DEVICE_SPACE uint *ray_queue,
    LUISA_DEVICE_SPACE uint &ray_queue_size,
    LUISA_UNIFORM_SPACE PrepareForTileUniforms &uniforms,
    uint tid) noexcept {
    
    auto ray_count = uniforms.viewport.z * uniforms.viewport.w;
    
    if (tid < ray_count) {
        
        auto dx = tid % uniforms.viewport.z;
        auto dy = tid / uniforms.viewport.z;
        
        auto x = uniforms.viewport.x + dx;
        auto y = uniforms.viewport.y + dy;
        
        ray_queue[tid] = y * uniforms.resolution.x + x;
        
        if (tid == 0u) { ray_queue_size = ray_count; }
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "viewport.h"
#include "parser.h"
#include "scene.h"
#include "camera.h"
#include "sampler.h"

namespace luisa {

class Integrator : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Integrator);

protected:
    Scene *_scene{nullptr};
    Camera *_camera{nullptr};
    Sampler *_sampler{nullptr};
    Viewport _viewport{};
    
    virtual void _prepare_for_frame() = 0;

public:
    Integrator(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) noexcept
        : Node{device} {}
    
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

#endif
