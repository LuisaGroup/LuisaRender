//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <optional>
#include <luisa-compute.h>

#include <scene/shape.h>
#include <scene/light.h>
#include <scene/camera.h>
#include <scene/film.h>
#include <scene/filter.h>
#include <scene/material.h>
#include <scene/transform.h>
#include <scene/integrator.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelBuildHint;
using compute::BindlessArray;
using compute::BindlessBuffer;
using compute::BindlessTexture2D;
using compute::BindlessTexture3D;
using compute::Buffer;
using compute::BufferView;
using compute::Callable;
using compute::Device;
using compute::Hit;
using compute::Image;
using compute::Mesh;
using compute::PixelStorage;
using compute::Resource;
using compute::Triangle;
using compute::Volume;

class Scene;

class Pipeline {

public:
    template<typename T, uint buffer_id_shift, uint buffer_element_alignment>
    class BufferArena {

    public:
        static constexpr auto buffer_capacity = (1u << buffer_id_shift) * buffer_element_alignment;

    private:
        Pipeline &_pipeline;
        Buffer<T> *_buffer{nullptr};
        uint _buffer_id{0u};
        uint _buffer_offset{0u};

    public:
        explicit BufferArena(Pipeline &pipeline) noexcept : _pipeline{pipeline} {}
        [[nodiscard]] std::pair<BufferView<T>, uint /* buffer id and offset */> allocate(size_t n) noexcept;
    };

public:
    static constexpr size_t bindless_array_capacity = 500'000u;// limitation of Metal
    using arena_buffer_block_type = uint4;
    static constexpr size_t arena_buffer_block_size = sizeof(arena_buffer_block_type);
    static constexpr size_t arena_buffer_block_count = 4u * 1024u * 1024u;
    static constexpr size_t arena_buffer_size = arena_buffer_block_size * arena_buffer_block_count;// == 64MB
    static constexpr size_t arena_allocation_threshold = 4_mb;
    using ResourceHandle = luisa::unique_ptr<Resource>;

    struct MeshData {
        Mesh *resource;
        uint triangle_count;
        uint position_buffer_id_and_offset;
        uint attribute_buffer_id_and_offset;
        uint triangle_buffer_id_and_offset;
        uint area_cdf_buffer_id_and_offset;
    };

private:
    Device &_device;
    Accel _accel;
    TransformTree _transform_tree;
    BindlessArray _bindless_array;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    luisa::vector<ResourceHandle> _resources;
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    luisa::vector<luisa::unique_ptr<Material::Interface>> _material_interfaces;
    luisa::vector<luisa::unique_ptr<Light::Interface>> _light_interfaces;
    luisa::unordered_map<luisa::string /* impl type */, uint /* tag */, Hash64> _material_tags;
    luisa::unordered_map<luisa::string /* impl type */, uint /* tag */, Hash64> _light_tags;
    luisa::unordered_map<const Material *, std::pair<uint /* buffer id and tag */, uint /* properties */>> _materials;
    luisa::unordered_map<const Light *, std::pair<uint /* buffer id and tag */, uint /* properties */>> _lights;
    BufferArena<float3, MeshInstance::position_buffer_id_shift, MeshInstance::position_buffer_element_alignment> _position_buffer_arena;
    BufferArena<VertexAttribute, MeshInstance::attribute_buffer_id_shift, MeshInstance::attribute_buffer_element_alignment> _attribute_buffer_arena;
    BufferArena<Triangle, MeshInstance::triangle_buffer_id_shift, MeshInstance::triangle_buffer_element_alignment> _triangle_buffer_arena;
    BufferArena<float, MeshInstance::area_cdf_buffer_id_shift, MeshInstance::area_cdf_buffer_element_alignment> _area_cdf_buffer_arena;
    luisa::vector<MeshInstance> _instances;
    Buffer<MeshInstance> _instance_buffer;
    luisa::vector<luisa::unique_ptr<Camera::Instance>> _cameras;
    luisa::vector<luisa::unique_ptr<Filter::Instance>> _filters;
    luisa::vector<luisa::unique_ptr<Film::Instance>> _films;
    luisa::unique_ptr<Integrator::Instance> _integrator;
    luisa::unique_ptr<Sampler::Instance> _sampler;

private:
    void _build_geometry(CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes, float init_time, AccelBuildHint hint) noexcept;
    void _process_shape(
        CommandBuffer &command_buffer, TransformTree::Builder &transform_builder, const Shape *shape,
        const Material *overridden_material = nullptr, const Light *overridden_light = nullptr) noexcept;
    [[nodiscard]] std::pair<uint /* buffer id and tag */, uint /* property flags */> _process_material(CommandBuffer &command_buffer, const Material *material) noexcept;
    [[nodiscard]] std::pair<uint /* buffer id and tag */, uint /* property flags */> _process_light(CommandBuffer &command_buffer, const Shape *shape, const Light *light) noexcept;

public:
    // for internal use only; use Pipeline::create() instead
    explicit Pipeline(Device &device) noexcept;
    Pipeline(Pipeline &&) noexcept = delete;
    Pipeline(const Pipeline &) noexcept = delete;
    Pipeline &operator=(Pipeline &&) noexcept = delete;
    Pipeline &operator=(const Pipeline &) noexcept = delete;
    ~Pipeline() noexcept;

public:
    template<typename T>
    [[nodiscard]] auto register_bindless(BufferView<T> buffer) noexcept {
        auto buffer_id = _bindless_buffer_count++;
        _bindless_array.emplace(buffer_id, buffer);
        return static_cast<uint>(buffer_id);
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Buffer<T> &buffer) noexcept {
        return register_bindless(buffer.view());
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Image<T> &image) noexcept {
        auto tex2d_id = _bindless_tex2d_count++;
        _bindless_array.emplace(tex2d_id, image);
        return static_cast<uint>(tex2d_id);
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Volume<T> &volume) noexcept {
        auto tex3d_id = _bindless_tex3d_count++;
        _bindless_array.emplace(tex3d_id, volume);
        return static_cast<uint>(tex3d_id);
    }

    template<typename T, typename... Args>
        requires std::is_base_of_v<Resource, T>
    [[nodiscard]] auto create(Args &&...args) noexcept -> T * {
        auto resource = luisa::make_unique<T>(_device.create<T>(std::forward<Args>(args)...));
        auto p = resource.get();
        _resources.emplace_back(std::move(resource));
        return p;
    }

    [[nodiscard]] auto &device() noexcept { return _device; }
    [[nodiscard]] const auto &device() const noexcept { return _device; }

    template<typename T>
    [[nodiscard]] auto bindless_buffer(Expr<uint> buffer_id) const noexcept { return _bindless_array.buffer<T>(buffer_id); }
    [[nodiscard]] auto bindless_tex2d(Expr<uint> tex_id) const noexcept { return _bindless_array.tex2d(tex_id); }
    [[nodiscard]] auto bindless_tex3d(Expr<uint> tex_id) const noexcept { return _bindless_array.tex3d(tex_id); }

public:
    [[nodiscard]] static luisa::unique_ptr<Pipeline> create(Device &device, Stream &stream, const Scene &scene) noexcept;
    [[nodiscard]] auto &accel() const noexcept { return _accel; }
    [[nodiscard]] auto &bindless_array() const noexcept { return _bindless_array; }
    [[nodiscard]] auto &transform_tree() const noexcept { return _transform_tree; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer.view(); }
    [[nodiscard]] auto camera_count() const noexcept { return _cameras.size(); }
    [[nodiscard]] std::tuple<Camera::Instance *, Film::Instance *, Filter::Instance *> camera(size_t i) noexcept;
    [[nodiscard]] std::tuple<const Camera::Instance *, const Film::Instance *, const Filter::Instance *> camera(size_t i) const noexcept;
    [[nodiscard]] auto material_interfaces() const noexcept { return luisa::span{_material_interfaces}; }
    [[nodiscard]] auto light_interfaces() const noexcept { return luisa::span{_light_interfaces}; }
    [[nodiscard]] auto sampler() noexcept { return _sampler.get(); }
    [[nodiscard]] auto sampler() const noexcept { return _sampler.get(); }
    void update_geometry(CommandBuffer &command_buffer, float time) noexcept;
    void render(Stream &stream) noexcept;

    template<typename T, typename I>
    [[nodiscard]] auto buffer(I &&i) const noexcept { return _bindless_array.buffer<T>(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex2d(I &&i) const noexcept { return _bindless_array.tex2d(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex3d(I &&i) const noexcept { return _bindless_array.tex3d(std::forward<I>(i)); }

    [[nodiscard]] std::pair<Var<MeshInstance>, Var<float4x4>> instance(const Var<Hit> &hit) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Var<MeshInstance> &instance, const Var<Hit> &hit) const noexcept;
    [[nodiscard]] Var<float3> vertex_position(const Var<MeshInstance> &instance, const Var<Triangle> &triangle, const Var<Hit> &hit) const noexcept;
    [[nodiscard]] std::tuple<Var<float3> /* normal */, Var<float3> /* tangent */, Var<float2> /* uv */> vertex_attribute(
        const Var<MeshInstance> &instance, const Var<Triangle> &triangle, const Var<Hit> &hit) const noexcept;
};

}// namespace luisa::render
