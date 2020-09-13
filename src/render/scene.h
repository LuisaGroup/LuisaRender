//
// Created by Mike Smith on 2020/9/4.
//

#pragma once

#include <compute/device.h>
#include <compute/dispatcher.h>
#include <compute/buffer.h>
#include <compute/pipeline.h>
#include <compute/acceleration.h>

#include "shape.h"
#include "material.h"
#include "background.h"
#include "frame_data.h"

namespace luisa::render {

using compute::Device;
using compute::BufferView;
using compute::KernelView;
using compute::Dispatcher;
using compute::Acceleration;

using compute::Ray;
using compute::AnyHit;
using compute::ClosestHit;
using compute::EntityRange;

class Scene {

private:
    Device *_device;
    
    BufferView<float3> _positions;
    BufferView<float3> _normals;
    BufferView<float2> _tex_coords;
    BufferView<TriangleHandle> _triangles;
    BufferView<EntityHandle> _instance_entities;
    BufferView<uint> _instances;
    BufferView<float4x4> _instance_transforms;
    BufferView<DataBlock> _material_data;
    BufferView<MaterialHandle> _instance_materials;
    TransformTree _transform_tree;
    
    std::shared_ptr<Background> _background;
    
    std::unique_ptr<Acceleration> _acceleration;
    BufferView<AnyHit> _any_hit_buffer;
    BufferView<ClosestHit> _closest_hit_buffer;
    
    KernelView _retrieve_intersections_kernel;
    
    bool _is_static{false};
    float _time{};

private:
    void _update_geometry(Dispatcher &dispatch, float time);
    void _intersect_any(Dispatcher &dispatch, const BufferView<Ray> &rays, const BufferView<uint> &ray_count, const BufferView<AnyInteraction> &its);
    void _intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<uint> &ray_count_buffer, const InteractionBuffers &its_buffers);
    
    void _encode_geometry_buffers(const std::vector<std::shared_ptr<Shape>> &shapes,
                                  float3 *positions, float3 *normals, float2 *uvs,
                                  TriangleHandle *triangles,
                                  EntityHandle *entities,
                                  std::vector<EntityRange> &entity_ranges,  // (vertex offset, triangle offset, triangle count)
                                  std::vector<Material *> &instance_materials,
                                  uint *instances);
    
    void _process_geometry(const std::vector<std::shared_ptr<Shape>> &shapes);
    void _compile_retrieve_intersections_kernel();

public:
    Scene(Device *device,
          const std::vector<std::shared_ptr<Shape>> &shapes,
          std::shared_ptr<Background> background,
          float initial_time,
          size_t max_ray_count);
};

}


