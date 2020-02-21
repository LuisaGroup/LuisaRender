//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include "ray.h"
#include "hit.h"
#include "material.h"
#include "interaction.h"
#include "mathematics.h"

#ifndef LUISA_DEVICE_COMPATIBLE

#include "buffer.h"
#include "shape.h"
#include "light.h"
#include "geometry.h"
#include "acceleration.h"

namespace luisa {

class Scene : Noncopyable {

private:
    Geometry _geometry;

public:
    Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time)
        : _geometry{device, shapes, initial_time} {}
    
    void update(float time);
    
    static auto create(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time = 0.0f) {
        return std::make_unique<Scene>(device, shapes, lights, initial_time);
    }
    
    void trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<ClosestHit> hit_buffer);
    void trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<AnyHit> hit_buffer);
    void evaluate_interactions(KernelDispatcher &dispatch, BufferView<Ray> rays, BufferView<uint> ray_count, BufferView<ClosestHit> hits, InteractionBufferSet &interactions);
};

}

#endif
