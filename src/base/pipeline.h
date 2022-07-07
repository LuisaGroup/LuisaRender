//
// Created by Mike on 2021/12/15.
//

#pragma once

#include "core/hash.h"
#include <optional>

#include <luisa-compute.h>
#include <util/spec.h>
#include <base/shape.h>
#include <base/light.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/surface.h>
#include <base/transform.h>
#include <base/integrator.h>
#include <base/interaction.h>
#include <base/light_sampler.h>
#include <base/environment.h>
#include <base/texture.h>
#include <base/differentiation.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelUsageHint;
using compute::BindlessArray;
using compute::BindlessBuffer;
using compute::BindlessTexture2D;
using compute::BindlessTexture3D;
using compute::Buffer;
using compute::BufferArena;
using compute::BufferView;
using compute::Callable;
using compute::Device;
using compute::Hit;
using compute::Image;
using compute::Mesh;
using compute::PixelStorage;
using compute::Polymorphic;
using compute::Printer;
using compute::Ray;
using compute::Resource;
using compute::Triangle;
using compute::Volume;
using TextureSampler = compute::Sampler;

class Scene;

class Pipeline {

public:
    static constexpr auto bindless_array_capacity = 500'000u;// limitation of Metal
    static constexpr auto vertex_buffer_arena_size_elements = 1024u * 1024u;
    static constexpr auto transform_matrix_buffer_size = 1024u;

    using ResourceHandle = luisa::unique_ptr<Resource>;

    struct MeshGeometry {
        Mesh *resource;
        uint buffer_id_base;
    };

    struct MeshData {
        Mesh *resource;
        float shadow_term;
        uint geometry_buffer_id_base : 24;
        bool two_sided : 8;
    };

private:
    Device &_device;
    Accel _accel;
    TransformTree _transform_tree;
    BindlessArray _bindless_array;
    luisa::unique_ptr<BufferArena> _general_buffer_arena;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    luisa::vector<ResourceHandle> _resources;
    luisa::unordered_map<uint64_t, MeshGeometry> _mesh_cache;
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    Polymorphic<Surface::Instance> _surfaces;
    Polymorphic<Light::Instance> _lights;
    luisa::vector<Light::Handle> _instanced_lights;
    luisa::unordered_map<const Surface *, uint> _surface_tags;
    luisa::unordered_map<const Light *, uint> _light_tags;
    luisa::unordered_map<const Texture *, luisa::unique_ptr<Texture::Instance>> _textures;
    luisa::unordered_map<const Filter *, luisa::unique_ptr<Filter::Instance>> _filters;
    luisa::vector<Shape::Handle> _instances;
    luisa::vector<InstancedTransform> _dynamic_transforms;
    Buffer<Shape::Handle> _instance_buffer;
    luisa::vector<luisa::unique_ptr<Camera::Instance>> _cameras;
    luisa::unique_ptr<Spectrum::Instance> _spectrum;
    luisa::unique_ptr<Integrator::Instance> _integrator;
    luisa::unique_ptr<Environment::Instance> _environment;
    float _mean_time{};
    luisa::unique_ptr<Differentiation> _differentiation;
    // registered transforms
    luisa::unordered_map<const Transform *, uint> _transform_to_id;
    luisa::vector<const Transform *> _transforms;
    luisa::vector<float4x4> _transform_matrices;
    Buffer<float4x4> _transform_matrix_buffer;
    bool _any_dynamic_transforms{false};
    Printer _printer;

private:
    void _build_geometry(CommandBuffer &command_buffer,
                         luisa::span<const Shape *const> shapes,
                         float init_time, AccelUsageHint hint) noexcept;
    void _process_shape(CommandBuffer &command_buffer, const Shape *shape,
                        luisa::optional<bool> overridden_two_sided = luisa::nullopt,
                        const Surface *overridden_surface = nullptr,
                        const Light *overridden_light = nullptr) noexcept;
    [[nodiscard]] uint _process_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept;
    [[nodiscard]] uint _process_light(CommandBuffer &command_buffer, const Light *light) noexcept;

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
    [[nodiscard]] auto register_bindless(const Image<T> &image, TextureSampler sampler) noexcept {
        auto tex2d_id = _bindless_tex2d_count++;
        _bindless_array.emplace(tex2d_id, image, sampler);
        return static_cast<uint>(tex2d_id);
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Volume<T> &volume, TextureSampler sampler) noexcept {
        auto tex3d_id = _bindless_tex3d_count++;
        _bindless_array.emplace(tex3d_id, volume, sampler);
        return static_cast<uint>(tex3d_id);
    }

    void register_transform(const Transform *transform) noexcept;

    template<typename T, typename... Args>
        requires std::is_base_of_v<Resource, T>
    [[nodiscard]] auto create(Args &&...args) noexcept -> T* {
        auto resource = luisa::make_unique<T>(_device.create<T>(std::forward<Args>(args)...));
        auto p = resource.get();
        _resources.emplace_back(std::move(resource));
        return p;
    }

    template<typename T>
    [[nodiscard]] std::pair<BufferView<T>, uint /* bindless id */> arena_buffer(size_t n) noexcept {
        auto view = _general_buffer_arena->allocate<T>(
            std::max(n, static_cast<size_t>(1u)));
        auto buffer_id = register_bindless(view);
        return std::make_pair(view, buffer_id);
    }

    template<typename T>
    [[nodiscard]] auto bindless_buffer(Expr<uint> buffer_id) const noexcept { return _bindless_array.buffer<T>(buffer_id); }
    [[nodiscard]] auto bindless_tex2d(Expr<uint> tex_id) const noexcept { return _bindless_array.tex2d(tex_id); }
    [[nodiscard]] auto bindless_tex3d(Expr<uint> tex_id) const noexcept { return _bindless_array.tex3d(tex_id); }

public:
    [[nodiscard]] auto &device() const noexcept { return _device; }
    [[nodiscard]] static luisa::unique_ptr<Pipeline> create(
        Device &device, Stream &stream, const Scene &scene) noexcept;
    [[nodiscard]] auto &accel() const noexcept { return _accel; }
    [[nodiscard]] Differentiation &differentiation() noexcept;
    [[nodiscard]] const Differentiation &differentiation() const noexcept;
    [[nodiscard]] auto &bindless_array() noexcept { return _bindless_array; }
    [[nodiscard]] auto &bindless_array() const noexcept { return _bindless_array; }
    [[nodiscard]] auto &transform_tree() const noexcept { return _transform_tree; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer.view(); }
    [[nodiscard]] auto camera_count() const noexcept { return _cameras.size(); }
    [[nodiscard]] auto camera(size_t i) noexcept { return _cameras[i].get(); }
    [[nodiscard]] auto camera(size_t i) const noexcept { return _cameras[i].get(); }
    [[nodiscard]] auto &surfaces() const noexcept { return _surfaces; }
    [[nodiscard]] auto &lights() const noexcept { return _lights; }
    [[nodiscard]] auto instanced_lights() const noexcept { return luisa::span{_instanced_lights}; }
    [[nodiscard]] auto environment() const noexcept { return _environment.get(); }
    [[nodiscard]] auto integrator() const noexcept { return _integrator.get(); }
    [[nodiscard]] auto spectrum() const noexcept { return _spectrum.get(); }
    [[nodiscard]] auto has_lighting() const noexcept { return !_lights.empty() || _environment != nullptr; }
    [[nodiscard]] auto mean_time() const noexcept { return _mean_time; }
    [[nodiscard]] const Texture::Instance *build_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept;
    [[nodiscard]] const Filter::Instance *build_filter(CommandBuffer &command_buffer, const Filter *filter) noexcept;
    bool update(CommandBuffer &command_buffer, float time) noexcept;
    void render(Stream &stream) noexcept;
    [[nodiscard]] auto &printer() noexcept { return _printer; }

    template<typename T, typename I>
    [[nodiscard]] auto buffer(I &&i) const noexcept { return _bindless_array.buffer<T>(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex2d(I &&i) const noexcept { return _bindless_array.tex2d(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex3d(I &&i) const noexcept { return _bindless_array.tex3d(std::forward<I>(i)); }
    [[nodiscard]] Float4x4 transform(const Transform *transform) const noexcept;

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
