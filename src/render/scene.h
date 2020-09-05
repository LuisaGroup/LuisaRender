//
// Created by Mike Smith on 2020/9/4.
//

#pragma once

#include <compute/device.h>
#include <compute/dispatcher.h>
#include <compute/buffer.h>
#include <compute/acceleration.h>

#include "shape.h"
#include "material.h"
#include "background.h"
#include "pipeline.h"
#include "frame_data.h"

namespace luisa::render {

using compute::Device;
using compute::BufferView;
using compute::Dispatcher;
using compute::Acceleration;

using compute::Ray;
using compute::AnyHit;
using compute::ClosestHit;

class Scene {

public:
    struct Entity {
        uint vertex_offset;
        uint triangle_offset;
    };

private:
    Device *_device;
    
    BufferView<float3> _positions;
    BufferView<float3> _normals;
    BufferView<float2> _tex_coords;
    BufferView<packed_uint3> _triangles;
    BufferView<Entity> _instance_entities;
    BufferView<uint> _instances;
    BufferView<float4x4> _instance_transforms;
    BufferView<DataBlock> _material_data;
    BufferView<MaterialHandle> _instance_materials;
    TransformTree _transform_tree;
    std::shared_ptr<Background> _background;
    std::unique_ptr<Acceleration> _acceleration;
    BufferView<ClosestHit> _closest_hit_buffer;
    
    bool _is_static{false};
    float _time{};
    uint _ray_count{};

private:
    void _process(const std::vector<std::shared_ptr<Shape>> &shapes,
                  float3 *positions, float3 *normals, float2 *uvs,
                  packed_uint3 *triangles,
                  Entity *entities,
                  std::vector<packed_uint3> &entity_ranges,  // (vertex offset, triangle offset, triangle count)
                  std::vector<Material *> &instance_materials,
                  uint *instances);

public:
    Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, std::shared_ptr<Background> background, float initial_time, size_t max_ray_count);
    
    void update_geometry(Pipeline &pipeline, const float &time);
    void intersect_any(Pipeline &pipeline, const BufferView<Ray> &rays, const BufferView<uint> &ray_count, const BufferView<AnyInteraction> &its);
    void intersect_closest(Pipeline &pipeline, const BufferView<Ray> &ray_buffer, const BufferView<uint> &ray_count_buffer, const InteractionBuffers &its_buffers);
};

}

LUISA_STRUCT(luisa::render::Scene::Entity, vertex_offset, triangle_offset)
