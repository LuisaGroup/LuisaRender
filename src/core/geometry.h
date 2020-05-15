//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "hit.h"
#include "ray.h"
#include "mathematics.h"
#include "interaction.h"

namespace luisa::geometry {

struct EvaluateInteractionsKernelUniforms {
    uint attribute_flags;
    uint static_shape_light_begin;
    uint static_shape_light_end;
    uint dynamic_shape_light_begin;
    uint dynamic_shape_light_end;
    uint static_instance_light_begin;
    uint static_instance_light_end;
    uint dynamic_instance_light_begin;
    uint dynamic_instance_light_end;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <unordered_map>

#include "plugin.h"
#include "transform.h"
#include "acceleration.h"

namespace luisa {
class Geometry;
}

namespace luisa {

class GeometryEntity : Noncopyable {

private:
    Geometry *_geometry{nullptr};
    uint _vertex_offset{};
    uint _vertex_count{};
    uint _index_offset{};
    uint _index_count{};

public:
    GeometryEntity(Geometry *geometry, uint vertex_offset, uint vertex_count, uint index_offset, uint index_count)
        : _geometry{geometry}, _vertex_offset{vertex_offset}, _vertex_count{vertex_count}, _index_offset{index_offset}, _index_count{index_count} {}
    
    [[nodiscard]] BufferView<float3> position_buffer();
    [[nodiscard]] BufferView<float3> normal_buffer();
    [[nodiscard]] BufferView<float2> uv_buffer();
    [[nodiscard]] BufferView<packed_uint3> index_buffer();
    [[nodiscard]] uint triangle_count() const noexcept { return _index_count; }
    [[nodiscard]] uint triangle_offset() const noexcept { return _index_offset; }
    [[nodiscard]] uint vertex_offset() const noexcept { return _vertex_offset; }
    [[nodiscard]] uint vertex_count() const noexcept { return _vertex_count; }
};

class GeometryEncoder : Noncopyable {

private:
    friend class Geometry;
    [[nodiscard]] std::vector<float3> _steal_positions() noexcept { return std::move(_positions); }
    [[nodiscard]] std::vector<float3> _steal_normals() noexcept { return std::move(_normals); }
    [[nodiscard]] std::vector<float2> _steal_texture_coords() noexcept { return std::move(_tex_coords); }
    [[nodiscard]] std::vector<packed_uint3> _steal_indices() noexcept { return std::move(_indices); }

private:
    Geometry *_geometry;
    std::vector<float3> _positions;
    std::vector<float3> _normals;
    std::vector<float2> _tex_coords;
    std::vector<packed_uint3> _indices;
    uint _vertex_offset{0u};
    uint _index_offset{0u};

public:
    explicit GeometryEncoder(Geometry *geometry) noexcept: _geometry{geometry} {}
    void add_vertex(float3 position, float3 normal, float2 tex_coord) noexcept;
    void add_indices(uint3 indices) noexcept;
    void create(Shape *shape);
    void replicate(Shape *shape, Shape *reference);
    void instantiate(Shape *shape, Shape *reference);
    
};

class Geometry : Noncopyable {

public:
    friend class GeometryEncoder;
    friend class GeometryEntity;

private:
    Device *_device;
    std::vector<std::shared_ptr<Shape>> _static_shapes;
    std::vector<std::shared_ptr<Shape>> _static_instances;
    std::vector<std::shared_ptr<Shape>> _dynamic_shapes;
    std::vector<std::shared_ptr<Shape>> _dynamic_instances;
    std::unordered_map<Shape *, uint> _shape_to_instance_id;
    
    // for determining whether a light is hit
    uint _static_shape_light_begin{};
    uint _static_shape_light_end{};
    uint _dynamic_shape_light_begin{};
    uint _dynamic_shape_light_end{};
    uint _static_instance_light_begin{};
    uint _static_instance_light_end{};
    uint _dynamic_instance_light_begin{};
    uint _dynamic_instance_light_end{};
    
    // per-vertex attributes
    std::unique_ptr<Buffer<float3>> _position_buffer;
    std::unique_ptr<Buffer<float3>> _normal_buffer;
    std::unique_ptr<Buffer<float2>> _tex_coord_buffer;
    
    // per-face attributes
    std::unique_ptr<Buffer<packed_uint3>> _index_buffer;
    
    // per-instance attributes
    std::unique_ptr<Buffer<float4x4>> _dynamic_transform_buffer;
    std::unique_ptr<Buffer<uint>> _entity_index_buffer;
    std::unique_ptr<Buffer<uint>> _vertex_offset_buffer;
    std::unique_ptr<Buffer<uint>> _index_offset_buffer;
    
    std::unordered_map<Shape *, uint> _shape_to_entity_index;
    
    // acceleration
    std::unique_ptr<Acceleration> _acceleration;
    std::vector<std::unique_ptr<GeometryEntity>> _entities;
    
    // kernels
    std::unique_ptr<Kernel> _evaluate_interactions_kernel;

public:
    Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, const std::vector<std::shared_ptr<Light>> &lights, float initial_time);
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_shapes() const noexcept { return _static_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &static_instances() const noexcept { return _static_instances; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_shapes() const noexcept { return _dynamic_shapes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &dynamic_instances() const noexcept { return _dynamic_instances; }
    [[nodiscard]] const std::vector<std::unique_ptr<GeometryEntity>> &entities() const noexcept { return _entities; }
    [[nodiscard]] BufferView<float4x4> transform_buffer() { return _dynamic_transform_buffer->view(); }
    [[nodiscard]] BufferView<packed_uint3> index_buffer() { return _index_buffer->view(); }
    [[nodiscard]] BufferView<float3> position_buffer() { return _position_buffer->view(); }
    [[nodiscard]] BufferView<float3> normal_buffer() { return _normal_buffer->view(); }
    [[nodiscard]] BufferView<uint> entity_index_buffer() { return _entity_index_buffer->view(); }
    
    void update(float time);
    void trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<ClosestHit> hit_buffer);
    void trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<uint> ray_count, BufferView<AnyHit> hit_buffer);
    
    void evaluate_interactions(KernelDispatcher &dispatch,
                               BufferView<Ray> ray_buffer,
                               BufferView<uint> ray_count,
                               BufferView<ClosestHit> hit_buffer,
                               InteractionBufferSet &interaction_buffers);
    [[nodiscard]] uint entity_index(Shape *shape) const;
    [[nodiscard]] uint instance_index(Shape *shape) const;
    [[nodiscard]] uint instance_count() const noexcept;
    [[nodiscard]] GeometryEntity &entity(uint index);
};
    
}

#endif
