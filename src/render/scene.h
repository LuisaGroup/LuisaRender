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
#include "interaction.h"

namespace luisa::render {

struct LightSelection {
    uint index;
    float prob;
    ShaderSelection shader;
};

struct LightSample {
    float3 wi;
    float3 Li;
    float pdf;
    float distance;
};

}

LUISA_STRUCT(luisa::render::LightSelection, index, prob, shader)
LUISA_STRUCT(luisa::render::LightSample, wi, Li, pdf, distance)

namespace luisa::render {

using compute::Device;
using compute::BufferView;
using compute::KernelView;
using compute::Pipeline;
using compute::Acceleration;

using compute::Ray;
using compute::AnyHit;
using compute::ClosestHit;
using compute::MeshHandle;

class Scene {

private:
    Device *_device;
    
    BufferView<float3> _positions;
    BufferView<float3> _normals;
    BufferView<float2> _tex_coords;
    BufferView<TriangleHandle> _triangles;
    BufferView<float> _triangle_cdf_tables;
    BufferView<EntityHandle> _entities;
    BufferView<uint> _entity_triangle_counts;
    BufferView<uint> _instance_to_entity_id;
    BufferView<float4x4> _instance_transforms;
    
    TransformTree _transform_tree;
    
    // materials
    BufferView<MaterialHandle> _instance_materials;
    
    BufferView<float> _shader_weights;
    BufferView<float> _shader_cdf_tables;
    BufferView<uint> _shader_types;
    BufferView<uint> _shader_block_offsets;
    BufferView<DataBlock> _shader_blocks;
    
    // emitters
    BufferView<uint> _emitter_to_instance_id;
    BufferView<MaterialHandle> _emitter_materials;
    
    std::shared_ptr<Background> _background;
    
    std::unique_ptr<Acceleration> _acceleration;
    
    std::map<uint, SurfaceShader *> _surface_evaluate_functions;
    std::map<uint, SurfaceShader *> _surface_emission_functions;
    
    bool _is_static{false};

private:
    void _update_geometry(Pipeline &pipeline, float time);
    void _encode_geometry_buffers(const std::vector<std::shared_ptr<Shape>> &shapes,
                                  float3 *positions,
                                  float3 *normals,
                                  float2 *uvs,
                                  TriangleHandle *triangles,
                                  float *triangle_cdf_tables,
                                  EntityHandle *entities,
                                  std::vector<MeshHandle> &meshes,
                                  std::vector<Material *> &instance_materials,
                                  uint *instances);
    
    void _process_geometry(const std::vector<std::shared_ptr<Shape>> &instance_to_entity_id, float initial_time, std::vector<Material *> &instance_materials);
    void _process_materials(const std::vector<Material *> &shader_block_offsets);

public:
    Scene(Device *device,
          const std::vector<std::shared_ptr<Shape>> &shapes,
          std::shared_ptr<Background> background,
          float initial_time);
    
    [[nodiscard]] auto update_geometry(float time) {
        return [this, time](Pipeline &pipeline) { _update_geometry(pipeline, time); };
    }
    
    [[nodiscard]] auto intersect_any(const BufferView<Ray> &rays, BufferView<AnyHit> &hits) {
        return _acceleration->intersect_any(rays, hits);
    }
    
    [[nodiscard]] auto intersect_closest(const BufferView<Ray> &rays, BufferView<ClosestHit> &hits) {
        return _acceleration->intersect_closest(rays, hits);
    }
    
    [[nodiscard]] bool is_static() const noexcept { return _is_static; }
    [[nodiscard]] uint light_count() const noexcept { return _emitter_to_instance_id.size(); }
    
    [[nodiscard]] Expr<LightSelection> uniform_select_light(Expr<float> u_light, Expr<float> u_shader) const;
    [[nodiscard]] Expr<LightSample> uniform_sample_light(Expr<LightSelection> selection, Expr<float3> p, Expr<float2> u_shape) const;
    [[nodiscard]] Expr<Interaction> evaluate_interaction(Expr<Ray> ray, Expr<ClosestHit> hit, Expr<float> u_shader, uint flags) const;
    [[nodiscard]] Expr<Scattering> evaluate_scattering(Expr<Interaction> intr, Expr<float3> wi, Expr<float2> u, uint flags = SurfaceShader::EVAL_ALL);
};

}
