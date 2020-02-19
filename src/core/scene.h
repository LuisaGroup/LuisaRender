//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include "ray.h"
#include "hit.h"
#include "material.h"
#include "interaction.h"
#include "mathematics.h"

namespace luisa::scene {

LUISA_DEVICE_CALLABLE inline void evaluate_interactions(
    uint ray_count,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE const ClosestHit *hit_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const float2 *uv_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE const MaterialInfo *material_info_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_position_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_normal_buffer,
    LUISA_DEVICE_SPACE float2 *interaction_uv_buffer,
    LUISA_DEVICE_SPACE float4 *interaction_wo_and_distance_buffer,
    LUISA_DEVICE_SPACE MaterialInfo *interaction_material_info_buffer,
    uint attribute_flags,
    uint tid) noexcept {
    
    if (tid < ray_count) {
        
        auto hit = hit_buffer[tid];
        if (hit.distance <= 0.0f) { return; }
        
        auto indices = index_buffer[hit.triangle_index];
        
        using namespace interaction_attribute_flags;
        
        if ((attribute_flags & POSITION_BIT) || (attribute_flags & NORMAL_BIT) || (attribute_flags & WO_AND_DISTANCE_BIT)) {
            auto transform = transform_buffer[hit.instance_index];
            if (attribute_flags & NORMAL_BIT) {
                auto n = hit.bary_u * normal_buffer[indices.x] + hit.bary_v * normal_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * normal_buffer[indices.z];
//                interaction_normal_buffer[tid] = normalize(transpose(inverse(make_float3x3(transform))) * n);
                interaction_normal_buffer[tid] = n;
            }
            if ((attribute_flags & POSITION_BIT) || (attribute_flags & WO_AND_DISTANCE_BIT)) {
                auto p = hit.bary_u * position_buffer[indices.x] + hit.bary_v * position_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * position_buffer[indices.z];
                if (attribute_flags & POSITION_BIT) {
                    interaction_position_buffer[tid] = make_float3(transform * make_float4(p, 1.0f));
                }
                if (attribute_flags & WO_AND_DISTANCE_BIT) {
                    interaction_wo_and_distance_buffer[tid] = make_float4(normalize(make_float3(ray_buffer[tid].origin) - p), hit.distance);
                }
            }
        }
        
        if (attribute_flags & UV_BIT) {
            auto uv = hit.bary_u * uv_buffer[indices.x] + hit.bary_v * uv_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * uv_buffer[indices.z];
            interaction_uv_buffer[tid] = uv;
        }
        
        if (attribute_flags & MATERIAL_INFO_BIT) {
            interaction_material_info_buffer[tid] = material_info_buffer[hit.instance_index];
        }
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "buffer.h"
#include "shape.h"
#include "light.h"
#include "geometry.h"
#include "acceleration.h"

namespace luisa {
class GeometryEntity;
class GeometryEncoder;
}

namespace luisa {

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
    
    // per-vertex attributes
    std::unique_ptr<Buffer<float3>> _position_buffer;
    std::unique_ptr<Buffer<float3>> _normal_buffer;
    std::unique_ptr<Buffer<float2>> _tex_coord_buffer;
    
    // per-face attributes
    std::unique_ptr<Buffer<packed_uint3>> _index_buffer;
    
    // per-instance attributes
    std::unique_ptr<Buffer<float4x4>> _dynamic_transform_buffer;
    std::unique_ptr<Buffer<uint>> _entity_index_buffer;
    std::unique_ptr<Buffer<MaterialInfo>> _material_info_buffer;
    
    // light and material data buffers
    std::vector<std::unique_ptr<TypelessBuffer>> _light_data_buffers;
    std::vector<std::unique_ptr<TypelessBuffer>> _material_data_buffers;
    
    // acceleration
    std::unique_ptr<Acceleration> _acceleration;
    std::vector<std::unique_ptr<GeometryEntity>> _entities;
    
    // kernels
    std::unique_ptr<Kernel> _evaluate_interactions_kernel;

private:
    void _initialize_geometry(const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time);

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
    
    static std::unique_ptr<Scene> create(Device *device,
                                         const std::vector<std::shared_ptr<Shape>> &shapes,
                                         const std::vector<std::shared_ptr<Light>> &lights,
                                         float initial_time = 0.0f) { return std::make_unique<Scene>(device, shapes, lights, initial_time); }
    
    void trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<ClosestHit> hit_buffer);
    void trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<AnyHit> hit_buffer);
    
    void evaluate_interactions(KernelDispatcher &dispatch,
                               BufferView<Ray> ray_buffer,
                               BufferView<uint> ray_count,
                               BufferView<ClosestHit> hit_buffer,
                               InteractionBufferSet &interaction_buffers);
};

}

#endif
