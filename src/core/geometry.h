//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "data_types.h"
#include "ray.h"
#include "hit.h"
#include "interaction.h"

namespace luisa::geometry {

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

#include "node.h"
#include "acceleration.h"
#include "transform.h"
#include "interaction.h"

namespace luisa {
class Shape;
class GeometryEntity;
}

namespace luisa {

class Geometry : Noncopyable {

public:
    friend class GeometryEntity;
    friend class GeometryEncoder;

private:
    Device *_device;
    std::vector<std::shared_ptr<Shape>> _static_shapes;
    std::vector<std::shared_ptr<Shape>> _static_instances;
    std::vector<std::shared_ptr<Shape>> _dynamic_shapes;
    std::vector<std::shared_ptr<Shape>> _dynamic_instances;
    
    std::unique_ptr<Buffer> _position_buffer;
    std::unique_ptr<Buffer> _normal_buffer;
    std::unique_ptr<Buffer> _tex_coord_buffer;
    std::unique_ptr<Buffer> _index_buffer;
    std::unique_ptr<Buffer> _dynamic_transform_buffer;
    std::unique_ptr<Buffer> _entity_index_buffer;
    
    std::unique_ptr<Acceleration> _acceleration;
    std::vector<std::unique_ptr<GeometryEntity>> _entities;

public:
    Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time);
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_shapes() const noexcept { return _static_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_instances() const noexcept { return _static_instances; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_shapes() const noexcept { return _dynamic_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_instances() const noexcept { return _dynamic_instances; }
    [[nodiscard]] const std::vector<std::unique_ptr<GeometryEntity>> &entities() const noexcept { return _entities; }
    [[nodiscard]] BufferView<float4x4> transform_buffer() { return _dynamic_transform_buffer->view<float4x4>(); }
    [[nodiscard]] BufferView<uint> entity_index_buffer() { return _entity_index_buffer->view<uint>(); }
    void update(KernelDispatcher &dispatch, float time);
    
    static std::unique_ptr<Geometry> create(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time = 0.0f) {
        return std::make_unique<Geometry>(device, shapes, initial_time);
    }
    
    void closest_hit(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<ClosestHit> hit_buffer);
    void any_hit(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<AnyHit> hit_buffer);
    void evaluate_interactions(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<ClosestHit> hit_buffer, BufferView<Interaction> interaction_buffer);
};

class GeometryEntity : Noncopyable {

public:
    friend class GeometryEncoder;

private:
    Geometry *_geometry{nullptr};
    uint _vertex_offset{};
    uint _vertex_count{};
    uint _index_offset{};
    uint _index_count{};

public:
    GeometryEntity(Geometry *geometry, uint vertex_offset, uint vertex_count, uint index_offset, uint index_count)
        : _geometry{geometry}, _vertex_offset{vertex_offset}, _vertex_count{vertex_count}, _index_offset{index_offset}, _index_count{index_count} {}
    
    [[nodiscard]] BufferView<float3> position_buffer() { return _geometry->_position_buffer->view<float3>(_vertex_offset, _vertex_count); }
    [[nodiscard]] BufferView<float3> normal_buffer() { return _geometry->_normal_buffer->view<float3>(_vertex_offset, _vertex_count); }
    [[nodiscard]] BufferView<float2> texture_coord_buffer() { return _geometry->_tex_coord_buffer->view<float2>(_vertex_offset, _vertex_count); }
    [[nodiscard]] BufferView<packed_uint3> index_buffer() { return _geometry->_index_buffer->view<packed_uint3>(_index_offset, _index_count); }
    [[nodiscard]] uint triangle_count() const noexcept { return _index_count; }
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
    
    [[nodiscard]] uint create() {
        auto entity_index = static_cast<uint>(_geometry->_entities.size());
        _geometry->_entities.emplace_back(std::make_unique<GeometryEntity>(
            _geometry,
            _vertex_offset, static_cast<uint>(_positions.size() - _vertex_offset),
            _index_offset, (static_cast<uint>(_indices.size() - _index_offset))));
        _vertex_offset = static_cast<uint>(_positions.size());
        _index_offset = static_cast<uint>(_indices.size());
        return entity_index;
    }
    
    [[nodiscard]] uint replicate(uint reference_index, float4x4 static_transform) {
        LUISA_ERROR_IF_NOT(reference_index < _geometry->_entities.size(), "invalid reference entity index for replicating: ", reference_index);
        auto &&reference = *_geometry->_entities[reference_index];
        if (static_transform == math::identity()) {
            for (auto i = reference._vertex_offset; i < reference._vertex_offset + reference._vertex_count; i++) {
                add_vertex(_positions[i], _normals[i], _tex_coords[i]);
            }
        } else {
            auto normal_matrix = transpose(inverse(make_float3x3(static_transform)));
            for (auto i = reference._vertex_offset; i < reference._vertex_offset + reference._vertex_count; i++) {
                add_vertex(make_float3(static_transform * make_float4(_positions[i], 1.0f)),
                           normalize(normal_matrix * _normals[i]),
                           _tex_coords[i]);
            }
        }
        for (auto i = 0u; i < reference._index_count; i++) {
            add_indices(make_uint3(_indices[i + reference._index_offset]) - make_uint3(reference._vertex_offset));
        }
        return create();
    }
    
    [[nodiscard]] uint instantiate(uint reference_index) noexcept {
        LUISA_ERROR_IF_NOT(reference_index < _geometry->_entities.size(), "invalid reference entity index for instancing: ", reference_index);
        return reference_index;
    }
    
};
    
}

#endif