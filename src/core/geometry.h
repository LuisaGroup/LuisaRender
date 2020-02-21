//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "hit.h"
#include "ray.h"
#include "mathematics.h"
#include "interaction.h"

namespace luisa::geometry {

LUISA_DEVICE_CALLABLE inline void evaluate_interactions(
    uint ray_count,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE const ClosestHit *hit_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const float2 *uv_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const uint *vertex_offset_buffer,
    LUISA_DEVICE_SPACE const uint *index_offset_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE bool *interaction_valid_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_position_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_normal_buffer,
    LUISA_DEVICE_SPACE float2 *interaction_uv_buffer,
    LUISA_DEVICE_SPACE float4 *interaction_wo_and_distance_buffer,
    uint attribute_flags,
    uint tid) noexcept {
    
    if (tid < ray_count) {
        
        auto hit = hit_buffer[tid];
        if (hit.distance <= 0.0f) {
            interaction_valid_buffer[tid] = false;
            return;
        }
        
        interaction_valid_buffer[tid] = true;
        auto indices = index_buffer[hit.triangle_index + index_offset_buffer[hit.instance_index]] + vertex_offset_buffer[hit.instance_index];
        
        using namespace interaction_attribute_flags;
        
        if ((attribute_flags & POSITION_BIT) || (attribute_flags & NORMAL_BIT) || (attribute_flags & WO_AND_DISTANCE_BIT)) {
            auto transform = transform_buffer[hit.instance_index];
            if (attribute_flags & NORMAL_BIT) {
                auto n = hit.bary_u * normal_buffer[indices.x] + hit.bary_v * normal_buffer[indices.y] + (1.0f - hit.bary_u - hit.bary_v) * normal_buffer[indices.z];
                interaction_normal_buffer[tid] = normalize(transpose(inverse(make_float3x3(transform))) * n);
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
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "node.h"
#include "transform.h"
#include "material.h"
#include "acceleration.h"

namespace luisa {
class Geometry;
}

namespace luisa {

class GeometryEntity : Noncopyable {

private:
    Geometry *_geometry{nullptr};
    uint _vertex_offset{};
    uint _vertex_count{};
    uint _index_offset{};
    uint _index_count{};

public:
    GeometryEntity(Geometry *geometry, uint vertex_offset, uint vertex_count, uint index_offset, uint index_count)
        : _geometry{geometry}, _vertex_offset{vertex_offset}, _vertex_count{vertex_count}, _index_offset{index_offset}, _index_count{index_count} {}
    
    [[nodiscard]] BufferView<float3> position_buffer();
    [[nodiscard]] BufferView<float3> normal_buffer();
    [[nodiscard]] BufferView<float2> uv_buffer();
    [[nodiscard]] BufferView<packed_uint3> index_buffer();
    [[nodiscard]] uint triangle_count() const noexcept { return _index_count; }
    [[nodiscard]] uint triangle_offset() const noexcept { return _index_offset; }
    [[nodiscard]] uint vertex_offset() const noexcept { return _vertex_offset; }
    [[nodiscard]] uint vertex_count() const noexcept { return _vertex_count; }
};

class GeometryEncoder : Noncopyable {

private:
    Geometry *_geometry;
    std::vector<float3> _positions;
    std::vector<float3> _normals;
    std::vector<float2> _tex_coords;
    std::vector<packed_uint3> _indices;
    uint _vertex_offset{0u};
    uint _index_offset{0u};

public:
    explicit GeometryEncoder(Geometry *geometry) noexcept : _geometry{geometry} {}
    
    [[nodiscard]] std::vector<float3> steal_positions() noexcept { return std::move(_positions); }
    [[nodiscard]] std::vector<float3> steal_normals() noexcept { return std::move(_normals); }
    [[nodiscard]] std::vector<float2> steal_texture_coords() noexcept { return std::move(_tex_coords); }
    [[nodiscard]] std::vector<packed_uint3> steal_indices() noexcept { return std::move(_indices); }
    
    void add_vertex(float3 position, float3 normal, float2 tex_coord) noexcept {
        _positions.emplace_back(position);
        _normals.emplace_back(normal);
        _tex_coords.emplace_back(tex_coord);
    }
    
    void add_indices(uint3 indices) noexcept { _indices.emplace_back(make_packed_uint3(indices)); }
    
    [[nodiscard]] uint create();
    [[nodiscard]] uint replicate(uint reference_index, float4x4 static_transform);
    [[nodiscard]] uint instantiate(uint reference_index) noexcept;
    
};

class Geometry : Noncopyable {

private:
    friend class GeometryEncoder;
    friend class GeometryEntity;

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
    std::unique_ptr<Buffer<uint>> _vertex_offset_buffer;
    std::unique_ptr<Buffer<uint>> _index_offset_buffer;
    
    // acceleration
    std::unique_ptr<Acceleration> _acceleration;
    std::vector<std::unique_ptr<GeometryEntity>> _entities;
    
    // kernels
    std::unique_ptr<Kernel> _evaluate_interactions_kernel;
    
public:
    Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time);
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_shapes() const noexcept { return _static_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_instances() const noexcept { return _static_instances; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_shapes() const noexcept { return _dynamic_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_instances() const noexcept { return _dynamic_instances; }
    [[nodiscard]] const std::vector<std::unique_ptr<GeometryEntity>> &entities() const noexcept { return _entities; }
    [[nodiscard]] BufferView<float4x4> transform_buffer() { return _dynamic_transform_buffer->view(); }
    [[nodiscard]] BufferView<uint> entity_index_buffer() { return _entity_index_buffer->view(); }
    
    void update(float time);
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
