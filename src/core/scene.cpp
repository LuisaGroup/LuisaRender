//
// Created by Mike Smith on 2020/2/17.
//

#include "scene.h"

namespace luisa {

Scene::Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time) : _device{device} {
    
    LUISA_WARNING_IF(shapes.empty(), "no shape in scene");
    LUISA_WARNING_IF(lights.empty(), "no light in scene");
    
    for (auto &&shape : shapes) {
        if (shape->is_instance()) {
            if (shape->transform().is_static()) {
                _static_instances.emplace_back(shape)->unload();
            } else {
                _dynamic_instances.emplace_back(shape)->unload();
            }
        } else if (shape->transform().is_static()) {
            _static_shapes.emplace_back(shape)->unload();
        } else {
            _dynamic_shapes.emplace_back(shape)->unload();
        }
    }
    
    GeometryEncoder geometry_encoder{this};
    for (auto &&shape : _static_shapes) { if (!shape->loaded()) { shape->load(geometry_encoder); }}
    for (auto &&shape : _static_instances) { shape->load(geometry_encoder); }
    for (auto &&shape : _dynamic_shapes) { if (!shape->loaded()) { shape->load(geometry_encoder); }}
    for (auto &&shape : _dynamic_instances) { shape->load(geometry_encoder); }
    
    // create buffers
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
    
    _dynamic_transform_buffer = _device->create_buffer<float4x4>(shapes.size(), BufferStorage::MANAGED);
    _entity_index_buffer = _device->create_buffer<uint>(shapes.size(), BufferStorage::MANAGED);
    auto offset = 0u;
    auto transform_buffer = _dynamic_transform_buffer->view();
    auto entity_index_buffer = _entity_index_buffer->view();
    for (auto &&shape : _static_shapes) {
        transform_buffer[offset] = math::identity();
        entity_index_buffer[offset] = shape->entity_index();
        offset++;
    }
    for (auto &&shape : _static_instances) {
        transform_buffer[offset] = shape->transform().static_matrix();
        entity_index_buffer[offset] = shape->entity_index();
        offset++;
    }
    for (auto &&shape : _dynamic_shapes) {
        transform_buffer[offset] = shape->transform().dynamic_matrix(initial_time);
        entity_index_buffer[offset] = shape->entity_index();
        offset++;
    }
    for (auto &&shape : _dynamic_instances) {
        transform_buffer[offset++] = shape->transform().dynamic_matrix(initial_time) * shape->transform().static_matrix();
        entity_index_buffer[offset] = shape->entity_index();
        offset++;
    }
    
    transform_buffer.upload();
    entity_index_buffer.upload();
    
    _acceleration = _device->create_acceleration(*this);
}

void Scene::update(float time) {
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

}
