//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include "ray.h"
#include "hit.h"
#include "device.h"
#include "buffer.h"

namespace luisa::scene {

LUISA_DEVICE_CALLABLE inline void evaluate_interactions(
    uint ray_count,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE const ClosestHit *hit_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const float2 *tex_coord_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE const int16_t *material_id_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_position_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_normal_buffer,
    LUISA_DEVICE_SPACE float2 *interaction_uv_buffer,
    LUISA_DEVICE_SPACE float4 *interaction_wo_and_distance_buffer,
    LUISA_DEVICE_SPACE int16_t *interaction_material_id_buffer,
    uint tid) noexcept {
    
    if (tid < ray_count) {
        
        auto hit = hit_buffer[tid];
        if (hit.distance <= 0.0f) {
            return;
        }
        
        auto indices = index_buffer[hit.triangle_index];
        auto p = hit.bary_u * position_buffer[indices.x] + hit.bary_v * position_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * position_buffer[indices.z];
        auto n = hit.bary_u * normal_buffer[indices.x] + hit.bary_v * normal_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * normal_buffer[indices.z];
        auto uv = hit.bary_u * tex_coord_buffer[indices.x] + hit.bary_v * tex_coord_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * tex_coord_buffer[indices.z];
        
        auto transform = transform_buffer[hit.instance_index];
        
        interaction_position_buffer[tid] = make_float3(transform * make_float4(p, 1.0f));
        interaction_normal_buffer[tid] = normalize(transpose(inverse(make_float3x3(transform))) * n);
        interaction_uv_buffer[tid] = uv;
        interaction_wo_and_distance_buffer[tid] = make_float4(normalize(make_float3(ray_buffer[tid].origin) - p), hit.distance);
        interaction_material_id_buffer[tid] = material_id_buffer[hit.instance_index];
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "device.h"
#include "shape.h"
#include "light.h"
#include "geometry.h"

namespace luisa {

struct InteractionBufferSetView {
    BufferView<float3> position_buffer;
    BufferView<float3> normal_buffer;
    BufferView<float2> uv_buffer;
    BufferView<uint8_t> material_tag_buffer;
    BufferView<uint> material_index_buffer;
    BufferView<float4> wo_and_distance_buffer;
};

struct InteractionBufferSet {
    
    std::unique_ptr<Buffer<float3>> position_buffer;
    std::unique_ptr<Buffer<float3>> normal_buffer;
    std::unique_ptr<Buffer<float2>> uv_buffer;
    std::unique_ptr<Buffer<uint8_t>> material_tag_buffer;
    std::unique_ptr<Buffer<uint>> material_index_buffer;
    std::unique_ptr<Buffer<float4>> wo_and_distance_buffer;
    
    InteractionBufferSet(Device *device, size_t capacity)
        : position_buffer{device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE)},
          normal_buffer{device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE)},
          uv_buffer{device->create_buffer<float2>(capacity, BufferStorage::DEVICE_PRIVATE)},
          material_tag_buffer{device->create_buffer<uint8_t>(capacity, BufferStorage::DEVICE_PRIVATE)},
          material_index_buffer{device->create_buffer<uint>(capacity, BufferStorage::DEVICE_PRIVATE)},
          wo_and_distance_buffer{device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE)} {}
    
    [[nodiscard]] auto view() {
        return InteractionBufferSetView{
            position_buffer->view(),
            normal_buffer->view(),
            uv_buffer->view(),
            material_tag_buffer->view(),
            material_index_buffer->view(),
            wo_and_distance_buffer->view()};
    }
    
};

class Scene : Noncopyable {

public:
    friend class GeometryEntity;
    friend class GeometryEncoder;

private:
    Device *_device;
    std::vector<std::shared_ptr<Shape>> _static_shapes;
    std::vector<std::shared_ptr<Shape>> _static_instances;
    std::vector<std::shared_ptr<Shape>> _dynamic_shapes;
    std::vector<std::shared_ptr<Shape>> _dynamic_instances;
    
    std::unique_ptr<Buffer<float3>> _position_buffer;
    std::unique_ptr<Buffer<float3>> _normal_buffer;
    std::unique_ptr<Buffer<float2>> _tex_coord_buffer;
    std::unique_ptr<Buffer<packed_uint3>> _index_buffer;
    std::unique_ptr<Buffer<float4x4>> _dynamic_transform_buffer;
    std::unique_ptr<Buffer<uint>> _entity_index_buffer;
    
    std::unique_ptr<Acceleration> _acceleration;
    std::vector<std::unique_ptr<GeometryEntity>> _entities;

public:
    Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time);
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_shapes() const noexcept { return _static_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_instances() const noexcept { return _static_instances; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_shapes() const noexcept { return _dynamic_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_instances() const noexcept { return _dynamic_instances; }
    [[nodiscard]] const std::vector<std::unique_ptr<GeometryEntity>> &entities() const noexcept { return _entities; }
    [[nodiscard]] BufferView<float4x4> transform_buffer() { return _dynamic_transform_buffer->view(); }
    [[nodiscard]] BufferView<uint> entity_index_buffer() { return _entity_index_buffer->view(); }
    void update(float time);
    
    static std::unique_ptr<Scene> create(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time = 0.0f) {
        return std::make_unique<Scene>(device, shapes, lights, initial_time);
    }
    
    void shade(KernelDispatcher &dispatch,
               BufferView<Ray> ray_buffer,
               BufferView<float3> throughput_buffer,
               BufferView<float3> radiance_buffer,
               BufferView<uint8_t> depth_buffer);
};

}

#endif
