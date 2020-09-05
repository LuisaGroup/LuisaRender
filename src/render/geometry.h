//
// Created by Mike Smith on 2020/9/4.
//

#pragma once

#include <compute/device.h>
#include <compute/dispatcher.h>
#include <compute/buffer.h>
#include <compute/acceleration.h>

#include "shape.h"

namespace luisa::render {

using compute::Device;
using compute::BufferView;
using compute::Dispatcher;
using compute::Acceleration;

class TransformTree {

private:
    const Transform *_transform{nullptr};  // nullptr indicates the root of the tree
    std::vector<std::unique_ptr<TransformTree>> _children;
    uint _buffer_index{};

public:
    [[nodiscard]] TransformTree *add_child(const Transform *transform, uint buffer_index) {
        auto child = std::make_unique<TransformTree>();
        child->_transform = transform;
        child->_buffer_index = buffer_index;
        auto child_ptr = child.get();
        _children.emplace_back(std::move(child));
        return child_ptr;
    }
    
    void update(float4x4 *buffer, float time, float4x4 parent_matrix = make_float4x4(1.0f)) {
        if (_transform == nullptr) {  // top level
            for (auto &&child : _children) { update(buffer, time); }
        } else {
            auto m = parent_matrix * _transform->matrix(time);
            buffer[_buffer_index] = m;
            for (auto &&child : _children) { update(buffer, time, m); }
        }
    }
};

class Geometry {

public:
    struct Entity {
        uint triangle_offset;
        uint vertex_offset;
    };

private:
    BufferView<float3> _positions;
    BufferView<float3> _normals;
    BufferView<float2> _tex_coords;
    BufferView<packed_uint3> _triangles;
    BufferView<Entity> _entities;  // (index offset, vertex offset)
    BufferView<uint> _instances;  // indices into entities
    BufferView<float4x4> _instance_transforms;
    TransformTree _transform_tree;
    std::unordered_map<Shape *, uint> _shape_to_instance_id;
    
    std::unique_ptr<Acceleration> _acceleration;

private:
    void _encode(const std::vector<std::shared_ptr<Shape>> &shapes,
                 float3 *positions, float3 *normals, float2 *uvs,
                 packed_uint3 *triangles,
                 Entity *entities,
                 std::vector<packed_uint3> &entity_ranges,  // (vertex offset, triangle offset, triangle count)
                 uint *instances, float4x4 *instance_transforms);

public:
    Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes);
    
    [[nodiscard]] auto update(float time) {
        return _instance_transforms.modify([time, this](float4x4 *matrices) {
            _transform_tree.update(matrices, time);
        });
    }
};

}

LUISA_STRUCT(luisa::render::Geometry::Entity, triangle_offset, vertex_offset)
