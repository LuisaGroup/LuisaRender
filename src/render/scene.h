//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include "ray.h"
#include "hit.h"
#include "material.h"
#include "interaction.h"
#include <compute/mathematics.h>

#ifndef LUISA_DEVICE_COMPATIBLE

#include <compute/buffer.h>
#include "shape.h"
#include "light.h"
#include "geometry.h"
#include "illumination.h"
#include "acceleration.h"

namespace luisa {

class Scene : Noncopyable {

private:
    std::unique_ptr<Geometry> _geometry;
    std::unique_ptr<Illumination> _illumination;

public:
    Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time);
    
    void update(float time);
    
    static auto create(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time = 0.0f) {
        return std::make_unique<Scene>(device, shapes, lights, initial_time);
    }
    
    void trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<ClosestHit> hit_buffer);
    void trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<AnyHit> hit_buffer);
    void evaluate_interactions(KernelDispatcher &dispatch, BufferView<Ray> rays, BufferView<uint> ray_count, BufferView<ClosestHit> hits, InteractionBufferSet &interactions);
    [[nodiscard]] uint light_tag_count() const noexcept { return _illumination->tag_count(); }
};

}

#endif
