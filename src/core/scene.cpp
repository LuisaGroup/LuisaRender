//
// Created by Mike Smith on 2020/2/17.
//

#include "scene.h"
#include "geometry.h"

namespace luisa {

void Scene::update(float time) {
    _geometry->update(time);
}

void Scene::evaluate_interactions(KernelDispatcher &dispatch, BufferView<Ray> rays, BufferView<uint> ray_count, BufferView<ClosestHit> hits, InteractionBufferSet &interactions) {
    _geometry->evaluate_interactions(dispatch, rays, ray_count, hits, interactions);
}

void Scene::trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<ClosestHit> hit_buffer) {
    _geometry->trace_closest(dispatch, ray_buffer, ray_count, hit_buffer);
}

void Scene::trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<AnyHit> hit_buffer) {
    _geometry->trace_any(dispatch, ray_buffer, ray_count, hit_buffer);
}

Scene::Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time) {
    _geometry = std::make_unique<Geometry>(device, shapes, lights, initial_time);
    _illumination = std::make_unique<Illumination>(device, lights, _geometry.get());
}

}
