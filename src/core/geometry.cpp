//
// Created by Mike Smith on 2020/2/10.
//

#include "geometry.h"
#include "scene.h"

namespace luisa {

BufferView<float3> GeometryEntity::position_buffer() {
    return _scene->_position_buffer->view(_vertex_offset, _vertex_count);
}

BufferView<float3> GeometryEntity::normal_buffer() {
    return _scene->_normal_buffer->view(_vertex_offset, _vertex_count);
}

BufferView<float2> GeometryEntity::texture_coord_buffer() {
    return _scene->_tex_coord_buffer->view(_vertex_offset, _vertex_count);
}

BufferView<packed_uint3> GeometryEntity::index_buffer() {
    return _scene->_index_buffer->view(_index_offset, _index_count);
}

uint GeometryEncoder::create() {
    auto entity_index = static_cast<uint>(_scene->_entities.size());
    _scene->_entities.emplace_back(std::make_unique<GeometryEntity>(
        _scene,
        _vertex_offset, static_cast<uint>(_positions.size() - _vertex_offset),
        _index_offset, (static_cast<uint>(_indices.size() - _index_offset))));
    _vertex_offset = static_cast<uint>(_positions.size());
    _index_offset = static_cast<uint>(_indices.size());
    return entity_index;
}

uint GeometryEncoder::replicate(uint reference_index, float4x4 static_transform) {
    LUISA_ERROR_IF_NOT(reference_index < _scene->_entities.size(), "invalid reference entity index for replicating: ", reference_index);
    auto &&reference = *_scene->_entities[reference_index];
    auto normal_matrix = transpose(inverse(make_float3x3(static_transform)));
    for (auto i = reference._vertex_offset; i < reference._vertex_offset + reference._vertex_count; i++) {
        add_vertex(make_float3(static_transform * make_float4(_positions[i], 1.0f)),
                   normalize(normal_matrix * _normals[i]),
                   _tex_coords[i]);
    }
    for (auto i = 0u; i < reference._index_count; i++) {
        add_indices(make_uint3(_indices[i + reference._index_offset]));
    }
    return create();
}

uint GeometryEncoder::instantiate(uint reference_index) noexcept {
    LUISA_ERROR_IF_NOT(reference_index < _scene->_entities.size(), "invalid reference entity index for instancing: ", reference_index);
    return reference_index;
}

}
