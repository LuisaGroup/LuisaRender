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

void GeometryEncoder::create(Shape *shape) {
    auto entity_index = static_cast<uint>(_geometry->_entities.size());
    _geometry->_entities.emplace_back(std::make_unique<GeometryEntity>(
        _geometry,
        _vertex_offset, static_cast<uint>(_positions.size() - _vertex_offset),
        _index_offset, static_cast<uint>(_indices.size() - _index_offset)));
    _vertex_offset = static_cast<uint>(_positions.size());
    _index_offset = static_cast<uint>(_indices.size());
    LUISA_EXCEPTION_IF_NOT(_geometry->_shape_to_entity_index.count(shape) == 0u, "Recreating shape");
    _geometry->_shape_to_entity_index.emplace(shape, entity_index);
}

void GeometryEncoder::replicate(Shape *shape, Shape *reference) {
    
    LUISA_EXCEPTION_IF_NOT(_geometry->_shape_to_entity_index.count(shape) == 0u, "Recreating shape");
    LUISA_EXCEPTION_IF(shape == reference || reference->is_instance(), "Cannot replicate the shape itself or an instance");
    LUISA_EXCEPTION_IF_NOT(reference->transform().is_static(), "Only static shapes can be replicated");
    auto iter = _geometry->_shape_to_entity_index.find(reference);
    if (iter == _geometry->_shape_to_entity_index.end()) {
        reference->load(*this);
        iter = _geometry->_shape_to_entity_index.find(reference);
        LUISA_EXCEPTION_IF(iter == _geometry->_shape_to_entity_index.end(), "Reference shape not properly loaded");
    }
    
    LUISA_EXCEPTION_IF_NOT(_geometry->_shape_to_entity_index.count(shape) == 0u, "Recreating shape");
    LUISA_EXCEPTION_IF_NOT(_vertex_offset == _positions.size() && _index_offset == _indices.size(), "Adding vertices or indices before making a replica is not allowed");
    auto &&ref_entity = *_geometry->_entities[iter->second];
    auto static_transform = shape->transform().static_matrix();
    auto normal_matrix = transpose(inverse(make_float3x3(static_transform)));
    for (auto i = ref_entity.vertex_offset(); i < ref_entity.vertex_offset() + ref_entity.vertex_count(); i++) {
        add_vertex(make_float3(static_transform * make_float4(_positions[i], 1.0f)),
                   normalize(normal_matrix * _normals[i]),
                   _tex_coords[i]);
    }
    for (auto i = ref_entity.triangle_offset(); i < ref_entity.triangle_offset() + ref_entity.triangle_count(); i++) {
        add_indices(make_uint3(_indices[i]));
    }
    create(shape);
}

void GeometryEncoder::instantiate(Shape *shape, Shape *reference) {
    
    LUISA_EXCEPTION_IF_NOT(_geometry->_shape_to_entity_index.count(shape) == 0u, "Recreating shape");
    LUISA_EXCEPTION_IF(shape == reference || reference->is_instance(), "Cannot instantiate the shape itself or an instance");
    LUISA_EXCEPTION_IF_NOT(reference->transform().is_static(), "Only static shapes can be instantiate");
    
    auto iter = _geometry->_shape_to_entity_index.find(reference);
    if (iter == _geometry->_shape_to_entity_index.end()) {
        reference->load(*this);
        iter = _geometry->_shape_to_entity_index.find(reference);
        LUISA_EXCEPTION_IF(iter == _geometry->_shape_to_entity_index.end(), "Reference shape not properly loaded");
    }
    LUISA_EXCEPTION_IF_NOT(_vertex_offset == _positions.size() && _index_offset == _indices.size(), "Adding vertices or indices before making an instance is not allowed");
    _geometry->_shape_to_entity_index.emplace(shape, iter->second);
}

void GeometryEncoder::add_vertex(float3 position, float3 normal, float2 tex_coord) noexcept {
    _positions.emplace_back(position);
    _normals.emplace_back(normal);
    _tex_coords.emplace_back(tex_coord);
}

void GeometryEncoder::add_indices(uint3 indices) noexcept { _indices.emplace_back(make_packed_uint3(indices)); }

uint Geometry::entity_index(Shape *shape) const {
    auto iter = _shape_to_entity_index.find(shape);
    LUISA_EXCEPTION_IF(iter == _shape_to_entity_index.cend(), "Shape not loaded");
    return iter->second;
}

uint Geometry::instance_index(Shape *shape) const {
    auto iter = _shape_to_instance_id.find(shape);
    LUISA_EXCEPTION_IF(iter == _shape_to_instance_id.cend(), "Shape not found");
    return iter->second;
}

GeometryEntity &Geometry::entity(uint index) {
    return *_entities[index];
}

Geometry::Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time)
    : _device{device},
      _evaluate_interactions_kernel{device->load_kernel("geometry::evaluate_interactions")} {
    
    // load geometry
    LUISA_WARNING_IF(shapes.empty(), "No shape in scene");
    
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
    
    _static_shape_light_begin = _static_shapes.size();
    _static_instance_light_begin = _static_instances.size();
    _dynamic_shape_light_begin = _dynamic_shapes.size();
    _dynamic_instance_light_begin = _dynamic_instances.size();
    
    for (auto &&light : lights) {
        if (auto shape = light->shape(); shape != nullptr) {
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
    }
    
    _static_shape_light_end = _static_shapes.size();
    _static_instance_light_end = _static_instances.size();
    _dynamic_shape_light_end = _dynamic_shapes.size();
    _dynamic_instance_light_end = _dynamic_instances.size();
    
    // load shapes
    GeometryEncoder geometry_encoder{this};
    for (auto &&shape : _static_shapes) {
        if (_shape_to_entity_index.count(shape.get()) == 0u) {
            shape->load(geometry_encoder);
        }
    }
    for (auto &&shape : _static_instances) {
        shape->load(geometry_encoder);
    }
    for (auto &&shape : _dynamic_shapes) {
        if (_shape_to_entity_index.count(shape.get()) == 0u) {
            shape->load(geometry_encoder);
        }
    }
    for (auto &&shape : _dynamic_instances) {
        shape->load(geometry_encoder);
    }
    
    // create shape-to-instance_id map
    for (auto &&shape : _static_shapes) { _shape_to_instance_id.emplace(shape.get(), static_cast<uint>(_shape_to_instance_id.size())); }
    for (auto &&shape : _static_instances) { _shape_to_instance_id.emplace(shape.get(), static_cast<uint>(_shape_to_instance_id.size())); }
    for (auto &&shape : _dynamic_shapes) { _shape_to_instance_id.emplace(shape.get(), static_cast<uint>(_shape_to_instance_id.size())); }
    for (auto &&shape : _dynamic_instances) { _shape_to_instance_id.emplace(shape.get(), static_cast<uint>(_shape_to_instance_id.size())); }
    
    // create geometry buffers
    {
        auto positions = geometry_encoder._steal_positions();
        _position_buffer = _device->allocate_buffer<float3>(positions.size(), BufferStorage::MANAGED);
        std::copy(positions.cbegin(), positions.cend(), _position_buffer->view().data());
        _position_buffer->upload();
    }
    
    {
        auto normals = geometry_encoder._steal_normals();
        _normal_buffer = _device->allocate_buffer<float3>(normals.size(), BufferStorage::MANAGED);
        std::copy(normals.cbegin(), normals.cend(), _normal_buffer->view().data());
        _normal_buffer->upload();
    }
    
    {
        auto tex_coords = geometry_encoder._steal_texture_coords();
        _tex_coord_buffer = _device->allocate_buffer<float2>(tex_coords.size(), BufferStorage::MANAGED);
        std::copy(tex_coords.cbegin(), tex_coords.cend(), _tex_coord_buffer->view().data());
        _tex_coord_buffer->upload();
    }
    
    {
        auto indices = geometry_encoder._steal_indices();
        _index_buffer = _device->allocate_buffer<packed_uint3>(indices.size(), BufferStorage::MANAGED);
        std::copy(indices.cbegin(), indices.cend(), _index_buffer->view().data());
        _index_buffer->upload();
    }
    
    LUISA_INFO("Geometry loaded, vertices: ", _position_buffer->size(), ", triangles: ", _index_buffer->size());
    
    _dynamic_transform_buffer = _device->allocate_buffer<float4x4>(shapes.size(), BufferStorage::MANAGED);
    _entity_index_buffer = _device->allocate_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    _index_offset_buffer = _device->allocate_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    _vertex_offset_buffer = _device->allocate_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    
    auto offset = 0u;
    auto transform_buffer = _dynamic_transform_buffer->view();
    auto entity_index_buffer = _entity_index_buffer->view();
    auto index_offset_buffer = _index_offset_buffer->view();
    auto vertex_offset_buffer = _vertex_offset_buffer->view();
    
    for (auto &&shape : _static_shapes) {
        transform_buffer[offset] = math::identity();
        auto id = entity_index(shape.get());
        entity_index_buffer[offset] = id;
        index_offset_buffer[offset] = _entities[id]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[id]->vertex_offset();
        offset++;
    }
    for (auto &&shape : _static_instances) {
        auto id = entity_index(shape.get());
        transform_buffer[offset] = shape->transform().static_matrix();
        entity_index_buffer[offset] = id;
        index_offset_buffer[offset] = _entities[id]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[id]->vertex_offset();
        offset++;
    }
    for (auto &&shape : _dynamic_shapes) {
        auto id = entity_index(shape.get());
        transform_buffer[offset] = shape->transform().dynamic_matrix(initial_time);  // Note: dynamic shapes will not have null transforms
        entity_index_buffer[offset] = id;
        index_offset_buffer[offset] = _entities[id]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[id]->vertex_offset();
        offset++;
    }
    for (auto &&shape : _dynamic_instances) {
        auto id = entity_index(shape.get());
        transform_buffer[offset] = shape->transform().dynamic_matrix(initial_time) * shape->transform().static_matrix();  // Note: dynamic shapes will not have null transforms
        entity_index_buffer[offset] = id;
        index_offset_buffer[offset] = _entities[id]->triangle_offset();
        vertex_offset_buffer[offset] = _entities[id]->vertex_offset();
        offset++;
    }
    
    transform_buffer.upload();
    entity_index_buffer.upload();
    index_offset_buffer.upload();
    vertex_offset_buffer.upload();
    
    // create acceleration
    _acceleration = _device->build_acceleration(*this);
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

void Geometry::evaluate_interactions(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count,
                                     BufferView<ClosestHit> hit_buffer, InteractionBufferSet &interaction_buffers) {
    
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
        encode("interaction_state_buffer", interaction_buffers.state_buffer());
        if (interaction_buffers.has_position_buffer()) { encode("interaction_position_buffer", interaction_buffers.position_buffer()); }
        if (interaction_buffers.has_normal_buffer()) { encode("interaction_normal_buffer", interaction_buffers.normal_buffer()); }
        if (interaction_buffers.has_uv_buffer()) { encode("interaction_uv_buffer", interaction_buffers.uv_buffer()); }
        if (interaction_buffers.has_wo_and_pdf_buffer()) { encode("interaction_wo_and_pdf_buffer", interaction_buffers.wo_and_pdf_buffer()); }
        if (interaction_buffers.has_instance_id_buffer()) { encode("interaction_instance_id_buffer", interaction_buffers.instance_id_buffer()); }
        encode("uniforms", geometry::EvaluateInteractionsKernelUniforms{
            interaction_buffers.attribute_flags(),
            _static_shape_light_begin, _static_shape_light_end,
            _dynamic_shape_light_begin, _dynamic_shape_light_end,
            _static_instance_light_begin, _static_instance_light_end,
            _dynamic_instance_light_begin, _dynamic_instance_light_end});
    });
}

uint Geometry::instance_count() const noexcept {
    return _shape_to_instance_id.size();
}

}
