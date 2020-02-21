//
// Created by Mike Smith on 2020/2/10.
//

#include "geometry.h"
#include "scene.h"

namespace luisa {

BufferView<float3> GeometryEntity::position_buffer() {
    return _geometry->_position_buffer->view(_vertex_offset, _vertex_count);
}

BufferView<float3> GeometryEntity::normal_buffer() {
    return _geometry->_normal_buffer->view(_vertex_offset, _vertex_count);
}

BufferView<float2> GeometryEntity::uv_buffer() {
    return _geometry->_tex_coord_buffer->view(_vertex_offset, _vertex_count);
}

BufferView<packed_uint3> GeometryEntity::index_buffer() {
    return _geometry->_index_buffer->view(_index_offset, _index_count);
}

uint GeometryEncoder::create() {
    auto entity_index = static_cast<uint>(_geometry->_entities.size());
    _geometry->_entities.emplace_back(std::make_unique<GeometryEntity>(
        _geometry,
        _vertex_offset, static_cast<uint>(_positions.size() - _vertex_offset),
        _index_offset, (static_cast<uint>(_indices.size() - _index_offset))));
    _vertex_offset = static_cast<uint>(_positions.size());
    _index_offset = static_cast<uint>(_indices.size());
    return entity_index;
}

uint GeometryEncoder::replicate(uint reference_index, float4x4 static_transform) {
    LUISA_ERROR_IF_NOT(reference_index < _geometry->_entities.size(), "invalid reference entity index for replicating: ", reference_index);
    auto &&reference = *_geometry->_entities[reference_index];
    auto normal_matrix = transpose(inverse(make_float3x3(static_transform)));
    for (auto i = reference.vertex_offset(); i < reference.vertex_offset() + reference.vertex_count(); i++) {
        add_vertex(make_float3(static_transform * make_float4(_positions[i], 1.0f)),
                   normalize(normal_matrix * _normals[i]),
                   _tex_coords[i]);
    }
    for (auto i = 0u; i < reference.triangle_count(); i++) {
        add_indices(make_uint3(_indices[i + reference.triangle_offset()]));
    }
    return create();
}

uint GeometryEncoder::instantiate(uint reference_index) noexcept {
    LUISA_ERROR_IF_NOT(reference_index < _geometry->_entities.size(), "invalid reference entity index for instancing: ", reference_index);
    return reference_index;
}

Geometry::Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time)
    : _device{device},
      _evaluate_interactions_kernel{device->create_kernel("geometry_evaluate_interactions")} {
    
    // load geometry
    LUISA_WARNING_IF(shapes.empty(), "no shape in scene");
    
    for (auto &&shape : shapes) {
        if (shape->is_instance()) {
            if (shape->transform().is_static()) {
                _static_instances.emplace_back(shape);
            } else {
                _dynamic_instances.emplace_back(shape);
            }
        } else if (shape->transform().is_static()) {
            _static_shapes.emplace_back(shape);
        } else {
            _dynamic_shapes.emplace_back(shape);
        }
    }
    
    GeometryEncoder geometry_encoder{this};
    for (auto &&shape : _static_shapes) { if (!shape->loaded()) { shape->load(geometry_encoder); }}
    for (auto &&shape : _static_instances) { shape->load(geometry_encoder); }
    for (auto &&shape : _dynamic_shapes) { if (!shape->loaded()) { shape->load(geometry_encoder); }}
    for (auto &&shape : _dynamic_instances) { shape->load(geometry_encoder); }
    
    // create geometry buffers
    {
        auto positions = geometry_encoder.steal_positions();
        _position_buffer = _device->create_buffer<float3>(positions.size(), BufferStorage::MANAGED);
        std::copy(positions.cbegin(), positions.cend(), _position_buffer->view().data());
        _position_buffer->upload();
    }
    
    {
        auto normals = geometry_encoder.steal_normals();
        _normal_buffer = _device->create_buffer<float3>(normals.size(), BufferStorage::MANAGED);
        std::copy(normals.cbegin(), normals.cend(), _normal_buffer->view().data());
        _normal_buffer->upload();
    }
    
    {
        auto tex_coords = geometry_encoder.steal_texture_coords();
        _tex_coord_buffer = _device->create_buffer<float2>(tex_coords.size(), BufferStorage::MANAGED);
        std::copy(tex_coords.cbegin(), tex_coords.cend(), _tex_coord_buffer->view().data());
        _tex_coord_buffer->upload();
    }
    
    {
        auto indices = geometry_encoder.steal_indices();
        _index_buffer = _device->create_buffer<packed_uint3>(indices.size(), BufferStorage::MANAGED);
        std::copy(indices.cbegin(), indices.cend(), _index_buffer->view().data());
        _index_buffer->upload();
    }
    
    std::cout << "vertex count = " << _position_buffer->size() << std::endl;
    std::cout << "triangle count = " << _index_buffer->size() << std::endl;
    
    _dynamic_transform_buffer = _device->create_buffer<float4x4>(shapes.size(), BufferStorage::MANAGED);
    _entity_index_buffer = _device->create_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    _index_offset_buffer = _device->create_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    _vertex_offset_buffer = _device->create_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    
    auto offset = 0u;
    auto transform_buffer = _dynamic_transform_buffer->view();
    auto entity_index_buffer = _entity_index_buffer->view();
    auto index_offset_buffer = _index_offset_buffer->view();
    auto vertex_offset_buffer = _vertex_offset_buffer->view();
    
    for (auto &&shape : _static_shapes) {
        transform_buffer[offset] = math::identity();
        entity_index_buffer[offset] = shape->entity_index();
        index_offset_buffer[offset] = _entities[shape->entity_index()]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[shape->entity_index()]->vertex_offset();
        offset++;
    }
    for (auto &&shape : _static_instances) {
        transform_buffer[offset] = shape->transform().static_matrix();
        entity_index_buffer[offset] = shape->entity_index();
        index_offset_buffer[offset] = _entities[shape->entity_index()]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[shape->entity_index()]->vertex_offset();
        offset++;
    }
    for (auto &&shape : _dynamic_shapes) {
        transform_buffer[offset] = shape->transform().dynamic_matrix(initial_time);
        entity_index_buffer[offset] = shape->entity_index();
        index_offset_buffer[offset] = _entities[shape->entity_index()]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[shape->entity_index()]->vertex_offset();
        offset++;
    }
    for (auto &&shape : _dynamic_instances) {
        transform_buffer[offset] = shape->transform().dynamic_matrix(initial_time) * shape->transform().static_matrix();
        entity_index_buffer[offset] = shape->entity_index();
        index_offset_buffer[offset] = _entities[shape->entity_index()]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[shape->entity_index()]->vertex_offset();
        offset++;
    }
    
    transform_buffer.upload();
    entity_index_buffer.upload();
    index_offset_buffer.upload();
    vertex_offset_buffer.upload();
    
    // create acceleration
    _acceleration = _device->create_acceleration(*this);
}

void Geometry::update(float time) {
    
    if (!_dynamic_shapes.empty() || !_dynamic_instances.empty()) {
        auto dynamic_shape_offset = _static_shapes.size() + _static_instances.size();
        auto dynamic_instance_offset = dynamic_shape_offset + _dynamic_shapes.size();
        for (auto i = 0ul; i < _dynamic_shapes.size(); i++) {
            transform_buffer()[dynamic_shape_offset + i] = _dynamic_shapes[i]->transform().dynamic_matrix(time);
        }
        for (auto i = 0ul; i < _dynamic_instances.size(); i++) {
            transform_buffer()[dynamic_instance_offset + i] = _dynamic_instances[i]->transform().dynamic_matrix(time) * _dynamic_instances[i]->transform().static_matrix();
        }
        _dynamic_transform_buffer->view(dynamic_shape_offset, _dynamic_shapes.size() + _dynamic_instances.size()).upload();
        _device->launch_async([&](KernelDispatcher &dispatch) {
            _acceleration->refit(dispatch);
        });
    }
}

void Geometry::trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<ClosestHit> hit_buffer) {
    _acceleration->trace_closest(dispatch, ray_buffer, hit_buffer, ray_count);
}

void Geometry::trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<AnyHit> hit_buffer) {
    _acceleration->trace_any(dispatch, ray_buffer, hit_buffer, ray_count);
}

void Geometry::evaluate_interactions(KernelDispatcher &dispatch,
                                     BufferView<Ray> ray_buffer,
                                     BufferView<uint> ray_count,
                                     BufferView<ClosestHit> hit_buffer,
                                     InteractionBufferSet &interaction_buffers) {
    
    dispatch(*_evaluate_interactions_kernel, ray_buffer.size(), [&](KernelArgumentEncoder &encode) {
        encode("ray_buffer", ray_buffer);
        encode("ray_count", ray_count);
        encode("hit_buffer", hit_buffer);
        encode("position_buffer", *_position_buffer);
        encode("normal_buffer", *_normal_buffer);
        encode("uv_buffer", *_tex_coord_buffer);
        encode("index_buffer", *_index_buffer);
        encode("vertex_offset_buffer", *_vertex_offset_buffer);
        encode("index_offset_buffer", *_index_offset_buffer);
        encode("transform_buffer", *_dynamic_transform_buffer);
        encode("interaction_valid_buffer", interaction_buffers.valid_buffer());
        if (interaction_buffers.has_position_buffer()) { encode("interaction_position_buffer", interaction_buffers.position_buffer()); }
        if (interaction_buffers.has_normal_buffer()) { encode("interaction_normal_buffer", interaction_buffers.normal_buffer()); }
        if (interaction_buffers.has_uv_buffer()) { encode("interaction_uv_buffer", interaction_buffers.uv_buffer()); }
        if (interaction_buffers.has_wo_and_distance_buffer()) { encode("interaction_wo_and_distance_buffer", interaction_buffers.wo_and_distance_buffer()); }
        encode("attribute_flags", interaction_buffers.attribute_flags());
    });
    
}

}
