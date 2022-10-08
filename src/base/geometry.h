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
using compute::Mesh;
using compute::Var;
using compute::Expr;
using compute::Hit;
using compute::Float4x4;

class Pipeline;

class Geometry {

public:
    struct MeshGeometry {
        Mesh *resource;
        uint buffer_id_base;
    };

    struct MeshData {
        Mesh *resource;
        float shadow_term;
        uint geometry_buffer_id_base;
    };

private:
    Pipeline &_pipeline;
    Accel _accel;
    TransformTree _transform_tree;
    luisa::unordered_map<uint64_t, MeshGeometry> _mesh_cache;
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    luisa::vector<Light::Handle> _instanced_lights;
    luisa::vector<Shape::Handle> _instances;
    luisa::vector<InstancedTransform> _dynamic_transforms;
    Buffer<Shape::Handle> _instance_buffer;

private:
    void _process_shape(CommandBuffer &command_buffer, const Shape *shape, float init_time,
                        const Surface *overridden_surface = nullptr,
                        const Light *overridden_light = nullptr) noexcept;

public:
    explicit Geometry(Pipeline &pipeline) noexcept : _pipeline{pipeline} {};
    void build(CommandBuffer &command_buffer,
               luisa::span<const Shape *const> shapes,
               float init_time, AccelUsageHint hint) noexcept;
    bool update(CommandBuffer &command_buffer, float time) noexcept;
    [[nodiscard]] auto light_instances() const noexcept { return luisa::span{_instanced_lights}; }
    [[nodiscard]] Var<Hit> trace_closest(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] luisa::unique_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept;
    [[nodiscard]] Var<Shape::Handle> instance(Expr<uint> index) const noexcept;
    [[nodiscard]] Float4x4 instance_to_world(Expr<uint> index) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Var<Shape::Handle> &instance, Expr<uint> index) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(
        const Var<Shape::Handle> &instance, const Var<Triangle> &triangle, const Var<float3> &bary,
        const Var<float4x4> &shape_to_world, const Var<float3x3> &shape_to_world_normal) const noexcept;
    [[nodiscard]] auto intersect(const Var<Ray> &ray) const noexcept { return interaction(ray, trace_closest(ray)); }
    [[nodiscard]] auto intersect_any(const Var<Ray> &ray) const noexcept { return trace_any(ray); }
};

}// namespace luisa::render
