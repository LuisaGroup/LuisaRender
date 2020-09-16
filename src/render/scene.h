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
using compute::Pipeline;
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
    BufferView<DataBlock> _material_data;  // TODO
    BufferView<MaterialHandle> _instance_materials;
    TransformTree _transform_tree;
    
    std::shared_ptr<Background> _background;
    
    std::unique_ptr<Acceleration> _acceleration;
    BufferView<AnyHit> _any_hit_buffer;
    BufferView<ClosestHit> _closest_hit_buffer;
    InteractionBuffers _interaction_buffers;
    
    bool _is_static{false};

private:
    void _update_geometry(Pipeline &pipeline, float time);
    void _intersect_any(Pipeline &pipeline, const BufferView<Ray> &rays);
    void _intersect_closest(Pipeline &pipeline, const BufferView<Ray> &ray_buffer);
    
    void _encode_geometry_buffers(const std::vector<std::shared_ptr<Shape>> &shapes,
                                  float3 *positions, float3 *normals, float2 *uvs,
                                  TriangleHandle *triangles,
                                  EntityHandle *entities,
                                  std::vector<EntityRange> &entity_ranges,  // (vertex offset, triangle offset, triangle count)
                                  std::vector<Material *> &instance_materials,
                                  uint *instances);
    
    void _process_geometry(const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time);

public:
    Scene(Device *device,
          const std::vector<std::shared_ptr<Shape>> &shapes,
          std::shared_ptr<Background> background,
          float initial_time);
    
    [[nodiscard]] const BufferView<AnyHit> &any_hit_buffer() const noexcept { return _any_hit_buffer; }
    [[nodiscard]] const InteractionBuffers &interaction_buffers() const noexcept { return _interaction_buffers; }
    
    [[nodiscard]] auto update_geometry(float time) {
        return [this, time](Pipeline &pipeline) { _update_geometry(pipeline, time); };
    }
    
    [[nodiscard]] auto intersect_any(const BufferView<Ray> &rays) {
        return [this, &rays](Pipeline &pipeline) { _intersect_any(pipeline, rays); };
    }
    
    [[nodiscard]] auto intersect_closest(const BufferView<Ray> &rays) {
        return [this, &rays](Pipeline &pipeline) { _intersect_closest(pipeline, rays); };
    }
};

}
