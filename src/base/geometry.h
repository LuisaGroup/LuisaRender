//
// Created by Mike Smith on 2022/9/14.
//

#pragma once

#include <luisa-compute.h>
#include <base/transform.h>
#include <base/light.h>
#include <base/shape.h>
#include <base/interaction.h>

namespace luisa::render {

using compute::Accel;
using compute::Buffer;
using compute::CommandBuffer;
using compute::Expr;
using compute::Float4x4;
using compute::Hit;
using compute::Mesh;
using compute::Var;

class Pipeline;

class Geometry {

public:
    struct MeshGeometry {
        Mesh *resource;
        uint buffer_id_base;
    };

    struct MeshData {
        Mesh *resource;
        uint16_t shadow_term;
        uint16_t intersection_offset;
        uint geometry_buffer_id_base : 22;
        uint vertex_properties : 10;
    };

    static_assert(sizeof(MeshData) == 16u);

private:
    Pipeline &_pipeline;
    Accel _accel;
    TransformTree _transform_tree;
    luisa::unordered_map<uint64_t, MeshGeometry> _mesh_cache;
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    luisa::vector<Light::Handle> _instanced_lights;
    luisa::vector<uint4> _instances;
    luisa::vector<InstancedTransform> _dynamic_transforms;
    Buffer<uint4> _instance_buffer;
    float3 _world_min;
    float3 _world_max;

private:
    void _process_shape(
        CommandBuffer &command_buffer, const Shape *shape, float init_time,
        const Surface *overridden_surface = nullptr,
        const Light *overridden_light = nullptr,
        const Medium *overridden_medium = nullptr,
        bool overridden_visible = true) noexcept;

public:
    explicit Geometry(Pipeline &pipeline) noexcept : _pipeline{pipeline} {};
    void build(CommandBuffer &command_buffer,
               luisa::span<const Shape *const> shapes,
               float init_time, AccelUsageHint hint) noexcept;
    bool update(CommandBuffer &command_buffer, float time) noexcept;
    [[nodiscard]] auto instances() const noexcept { return luisa::span{_instances}; }
    [[nodiscard]] auto light_instances() const noexcept { return luisa::span{_instanced_lights}; }
    [[nodiscard]] auto world_min() const noexcept { return _world_min; }
    [[nodiscard]] auto world_max() const noexcept { return _world_max; }
    [[nodiscard]] Var<Hit> trace_closest(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(Expr<uint> inst_id, Expr<uint> prim_id,
                                                             Expr<float3> bary, Expr<float3> wo) const noexcept;
    [[nodiscard]] Shape::Handle instance(Expr<uint> index) const noexcept;
    [[nodiscard]] Float4x4 instance_to_world(Expr<uint> index) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Shape::Handle &instance, Expr<uint> index) const noexcept;
    [[nodiscard]] GeometryAttribute geometry_point(const Shape::Handle &instance, const Var<Triangle> &triangle,
                                                   const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(const Shape::Handle &instance, const Var<Triangle> &triangle,
                                                 const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] auto intersect(const Var<Ray> &ray) const noexcept { return interaction(ray, trace_closest(ray)); }
    [[nodiscard]] auto intersect_any(const Var<Ray> &ray) const noexcept { return trace_any(ray); }
};

}// namespace luisa::render
