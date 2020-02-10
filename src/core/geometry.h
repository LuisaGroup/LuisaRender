//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "node.h"
#include "acceleration.h"
#include "transform.h"

namespace luisa {
class Shape;
}

namespace luisa {

class Geometry {

public:
    friend class GeometryView;
    friend class GeometryEncoder;

private:
    Device *_device;
    std::vector<std::shared_ptr<Shape>> _shapes;
    std::unique_ptr<Buffer> _position_buffer;
    std::unique_ptr<Buffer> _normal_buffer;
    std::unique_ptr<Buffer> _tex_coord_buffer;
    std::unique_ptr<Buffer> _index_buffer;
    std::unique_ptr<Buffer> _dynamic_transform_buffer;
    std::unique_ptr<Acceleration> _acceleration;

public:
    Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes);
};

class GeometryView {

public:
    friend class GeometryEncoder;

private:
    Geometry *_geometry{nullptr};
    uint _vertex_offset{};
    uint _vertex_count{};
    uint _index_offset{};
    uint _index_count{};

public:
    GeometryView() noexcept = default;
    GeometryView(Geometry *geometry, uint vertex_offset, uint vertex_count, uint index_offset, uint index_count)
        : _geometry{geometry}, _vertex_offset{vertex_offset}, _vertex_count{vertex_count}, _index_offset{index_offset}, _index_count{index_count} {}
    
    [[nodiscard]] bool valid() const noexcept { return _geometry != nullptr; }
    void invalidate() noexcept { _geometry = nullptr; }
    [[nodiscard]] BufferView<float3> position_buffer() { return _geometry->_position_buffer->view<float3>(_vertex_offset, _vertex_count); }
    [[nodiscard]] BufferView<float3> normal_buffer() { return _geometry->_normal_buffer->view<float3>(_vertex_offset, _vertex_count); }
    [[nodiscard]] BufferView<float2> texture_coord_buffer() { return _geometry->_tex_coord_buffer->view<float2>(_vertex_offset, _vertex_count); }
    [[nodiscard]] BufferView<packed_uint3> index_buffer() { return _geometry->_index_buffer->view<packed_uint3>(_index_offset, _index_count); }
};

class GeometryEncoder {

private:
    Geometry *_geometry;
    std::vector<float3> &_positions;
    std::vector<float3> &_normals;
    std::vector<float2> &_tex_coords;
    std::vector<packed_uint3> &_indices;
    uint _vertex_offset;
    uint _index_offset;

public:
    explicit GeometryEncoder(Geometry *geometry,
                             std::vector<float3> &positions,
                             std::vector<float3> &normals,
                             std::vector<float2> &tex_coords,
                             std::vector<packed_uint3> &indices) noexcept
        : _geometry{geometry},
          _positions{positions},
          _normals{normals},
          _tex_coords{tex_coords},
          _indices{indices},
          _vertex_offset{static_cast<uint>(positions.size())},
          _index_offset{static_cast<uint>(indices.size())} {
        
        LUISA_ERROR_IF_NOT(_vertex_offset == _normals.size() && _vertex_offset == _tex_coords.size(), "corrupt geometry");
    }
    
    void add_vertex(float3 position, float3 normal, float2 tex_coord) noexcept {
        _positions.emplace_back(position);
        _normals.emplace_back(normal);
        _tex_coords.emplace_back(tex_coord);
    }
    
    void add_indices(uint3 indices) noexcept { _indices.emplace_back(make_packed_uint3(indices)); }
    
    [[nodiscard]] GeometryView create() {
        GeometryView view{_geometry,
                          _vertex_offset, static_cast<uint>(_positions.size() - _vertex_offset),
                          _index_offset, (static_cast<uint>(_indices.size() - _index_offset))};
        _vertex_offset = static_cast<uint>(_positions.size());
        _index_offset = static_cast<uint>(_indices.size());
        return view;
    }
    
    [[nodiscard]] GeometryView replicate(GeometryView another_view, float4x4 static_transform) {
        LUISA_ERROR_IF_NOT(_geometry == another_view._geometry, "shapes are from different geometry group");
        auto normal_matrix = transpose(inverse(make_float3x3(static_transform)));
        for (auto i = another_view._vertex_offset; i < another_view._vertex_offset + another_view._vertex_count; i++) {
            add_vertex(make_float3(static_transform * make_float4(_positions[i], 1.0f)),
                       normalize(normal_matrix * _normals[i]),
                       _tex_coords[i]);
        }
        for (auto i = 0u; i < another_view._index_count; i++) {
            add_indices(make_uint3(_indices[i + another_view._index_offset]) - make_uint3(another_view._vertex_offset));
        }
        return create();
    }
    
    [[nodiscard]] GeometryView instantiate(GeometryView another_view) {
        LUISA_ERROR_IF_NOT(_geometry == another_view._geometry, "shapes are from different geometry group");
        return another_view;
    }
};
    
}