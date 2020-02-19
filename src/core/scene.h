//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include "ray.h"
#include "hit.h"
#include "material.h"

namespace luisa::scene {

namespace interaction_attribute_flags {
LUISA_CONSTANT_SPACE constexpr auto POSITION_BIT = 0x01u;
LUISA_CONSTANT_SPACE constexpr auto NORMAL_BIT = 0x02u;
LUISA_CONSTANT_SPACE constexpr auto UV_BIT = 0x04u;
LUISA_CONSTANT_SPACE constexpr auto MATERIAL_INFO_BIT = 0x08u;
LUISA_CONSTANT_SPACE constexpr auto WO_AND_DISTANCE_BIT = 0x10u;
}

LUISA_DEVICE_CALLABLE inline void evaluate_interactions(
    uint ray_count,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE const ClosestHit *hit_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const float2 *tex_coord_buffer,
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
                interaction_normal_buffer[tid] = normalize(transpose(inverse(make_float3x3(transform))) * n);
            }
            if ((attribute_flags & POSITION_BIT) || (attribute_flags & WO_AND_DISTANCE_BIT)) {
                auto p = hit.bary_u * position_buffer[indices.x] + hit.bary_v * position_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * position_buffer[indices.z];
                interaction_position_buffer[tid] = make_float3(transform * make_float4(p, 1.0f));
                if (attribute_flags & WO_AND_DISTANCE_BIT) {
                    interaction_wo_and_distance_buffer[tid] = make_float4(normalize(make_float3(ray_buffer[tid].origin) - p), hit.distance);
                }
            }
        }
        
        if (attribute_flags & UV_BIT) {
            auto uv = hit.bary_u * tex_coord_buffer[indices.x] + hit.bary_v * tex_coord_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * tex_coord_buffer[indices.z];
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
#include "device.h"
#include "shape.h"
#include "light.h"
#include "geometry.h"

namespace luisa {

class InteractionBufferSet {

private:
    size_t _size{0ul};
    uint _attribute_flags{0x0u};
    
    std::unique_ptr<Buffer<float3>> _position_buffer;
    std::unique_ptr<Buffer<float3>> _normal_buffer;
    std::unique_ptr<Buffer<float2>> _uv_buffer;
    std::unique_ptr<Buffer<MaterialInfo>> _material_info_buffer;
    std::unique_ptr<Buffer<float4>> _wo_and_distance_buffer;

public:
    InteractionBufferSet() noexcept = default;
    
    InteractionBufferSet(Device *device, size_t capacity, uint flags = 0xffu)
        : _size{capacity},
          _attribute_flags{flags},
          _position_buffer{(flags & scene::interaction_attribute_flags::POSITION_BIT) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _normal_buffer{(flags & scene::interaction_attribute_flags::NORMAL_BIT) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _uv_buffer{(flags & scene::interaction_attribute_flags::UV_BIT) ? device->create_buffer<float2>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _material_info_buffer{(flags & scene::interaction_attribute_flags::MATERIAL_INFO_BIT) ?
                                device->create_buffer<MaterialInfo>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _wo_and_distance_buffer{(flags & scene::interaction_attribute_flags::WO_AND_DISTANCE_BIT) ?
                                  device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr} {}
    
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto attribute_flags() const noexcept { return _attribute_flags; }
    
    [[nodiscard]] bool has_position_buffer() const noexcept { return (_attribute_flags & scene::interaction_attribute_flags::POSITION_BIT) != 0u; }
    [[nodiscard]] bool has_normal_buffer() const noexcept { return (_attribute_flags & scene::interaction_attribute_flags::NORMAL_BIT) != 0u; }
    [[nodiscard]] bool has_uv_buffer() const noexcept { return (_attribute_flags & scene::interaction_attribute_flags::UV_BIT) != 0u; }
    [[nodiscard]] bool has_material_info_buffer() const noexcept { return (_attribute_flags & scene::interaction_attribute_flags::MATERIAL_INFO_BIT) != 0u; }
    [[nodiscard]] bool has_wo_and_distance_buffer() const noexcept { return (_attribute_flags & scene::interaction_attribute_flags::WO_AND_DISTANCE_BIT) != 0u; }
    
    [[nodiscard]] auto position_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_position_buffer(), "no position buffer present");
        return _position_buffer->view();
    }
    
    [[nodiscard]] auto normal_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_normal_buffer(), "no normal buffer present");
        return _normal_buffer->view();
    }
    
    [[nodiscard]] auto uv_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_uv_buffer(), "no uv buffer present");
        return _uv_buffer->view();
    }
    
    [[nodiscard]] auto material_info_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_material_info_buffer(), "no material info buffer present");
        return _material_info_buffer->view();
    }
    
    [[nodiscard]] auto wo_and_distance_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_wo_and_distance_buffer(), "no wo and distance buffer present");
        return _wo_and_distance_buffer->view();
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
    
    static std::unique_ptr<Scene> create(Device *device,
                                         const std::vector<std::shared_ptr<Shape>> &shapes,
                                         const std::vector<std::shared_ptr<Light>> &lights,
                                         float initial_time = 0.0f) {
        return std::make_unique<Scene>(device, shapes, lights, initial_time);
    }
    
    
    
};

}

#endif
