//
// Created by Mike Smith on 2020/2/10.
//

#include "geometry.h"
#include "shape.h"

namespace luisa {

Geometry::Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes)
    : _device{device} {
    
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> tex_coords;
    std::vector<packed_uint3> indices;
    
    for (auto &&shape : shapes) { if (!shape->is_instance() && shape->transform().is_static()) { _shapes.emplace_back(shape); }}
    for (auto &&shape : shapes) { if (shape->is_instance() || !shape->transform().is_static()) { _shapes.emplace_back(shape); }}  // shapes that requires a instance acceleration structure
    for (auto &&shape : _shapes) { shape->geometry_view().invalidate(); }
    for (auto &&shape : _shapes) { if (!shape->loaded()) { shape->load(GeometryEncoder{this, positions, normals, tex_coords, indices}); }}
    
    // create buffers
    _position_buffer = _device->create_buffer<float3>(positions.size(), BufferStorage::MANAGED);
    std::copy(positions.cbegin(), positions.cend(), _position_buffer->view<float3>().data());
    _position_buffer->upload();
    positions.clear();
    positions.shrink_to_fit();
    
    _normal_buffer = _device->create_buffer<float3>(normals.size(), BufferStorage::MANAGED);
    std::copy(normals.cbegin(), normals.cend(), _normal_buffer->view<float3>().data());
    _normal_buffer->upload();
    normals.clear();
    normals.shrink_to_fit();
    
    _tex_coord_buffer = _device->create_buffer<float2>(tex_coords.size(), BufferStorage::MANAGED);
    std::copy(tex_coords.cbegin(), tex_coords.cend(), _tex_coord_buffer->view<float2>().data());
    _tex_coord_buffer->upload();
    tex_coords.clear();
    tex_coords.shrink_to_fit();
    
    _index_buffer = _device->create_buffer<packed_uint3>(indices.size(), BufferStorage::MANAGED);
    std::copy(indices.cbegin(), indices.cend(), _index_buffer->view<packed_uint3>().data());
    _index_buffer->upload();
    indices.clear();
    indices.shrink_to_fit();
    
    _dynamic_transform_buffer = _device->create_buffer<float4x4>(_shapes.size(), BufferStorage::MANAGED);
}

}
