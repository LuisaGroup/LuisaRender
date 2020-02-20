//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#ifndef LUISA_DEVICE_COMPATIBLE

#include "node.h"
#include "transform.h"

namespace luisa {
class Scene;
}

namespace luisa {

class GeometryEntity : Noncopyable {

public:
    friend class GeometryEncoder;

private:
    Scene *_scene{nullptr};
    uint _vertex_offset{};
    uint _vertex_count{};
    uint _index_offset{};
    uint _index_count{};

public:
    GeometryEntity(Scene *scene, uint vertex_offset, uint vertex_count, uint index_offset, uint index_count)
        : _scene{scene}, _vertex_offset{vertex_offset}, _vertex_count{vertex_count}, _index_offset{index_offset}, _index_count{index_count} {}
    
    [[nodiscard]] BufferView<float3> position_buffer();
    [[nodiscard]] BufferView<float3> normal_buffer();
    [[nodiscard]] BufferView<float2> texture_coord_buffer();
    [[nodiscard]] BufferView<packed_uint3> index_buffer();
    [[nodiscard]] uint triangle_count() const noexcept { return _index_count; }
    [[nodiscard]] uint triangle_offset() const noexcept { return _index_offset; }
    [[nodiscard]] uint vertex_offset() const noexcept { return _vertex_offset; }
};

class GeometryEncoder : Noncopyable {

private:
    Scene *_scene;
    std::vector<float3> _positions;
    std::vector<float3> _normals;
    std::vector<float2> _tex_coords;
    std::vector<packed_uint3> _indices;
    uint _vertex_offset{0u};
    uint _index_offset{0u};

public:
    explicit GeometryEncoder(Scene *geometry) noexcept : _scene{geometry} {}
    
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
    
}

#endif