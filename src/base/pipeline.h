//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <optional>

#include <luisa-compute.h>
#include <util/spectrum.h>
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

namespace luisa::render {

using compute::Accel;
using compute::AccelBuildHint;
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
using compute::Ray;
using compute::Resource;
using compute::Triangle;
using compute::Volume;
using TextureSampler = compute::Sampler;

class Scene;

class Pipeline {

public:
    static constexpr size_t bindless_array_capacity = 500'000u;// limitation of Metal
    static constexpr auto vertex_buffer_arena_size_elements = 1024u * 1024u;
    using ResourceHandle = luisa::unique_ptr<Resource>;

    struct MeshGeometry {
        Mesh *resource;
        uint buffer_id_base;
    };

    struct MeshData {
        Mesh *resource;
        uint geometry_buffer_id_base;
        uint alpha_texture_id;
        float alpha;
        bool two_sided;
        bool is_virtual;
    };

private:
    Device &_device;
    Accel _accel;
    TransformTree _transform_tree;
    BindlessArray _bindless_array;
    luisa::unique_ptr<BufferArena> _position_buffer_arena;
    luisa::unique_ptr<BufferArena> _attribute_buffer_arena;
    luisa::unique_ptr<BufferArena> _general_buffer_arena;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    luisa::vector<ResourceHandle> _resources;
    luisa::unordered_map<uint64_t, MeshGeometry> _mesh_cache;
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    luisa::vector<const Surface::Instance *> _surfaces;
    luisa::vector<const Light::Instance *> _lights;
    luisa::unordered_map<const Surface *, uint> _surface_tags;
    luisa::unordered_map<const Light *, uint> _light_tags;
    luisa::unordered_map<const Texture *, const Texture::Instance *> _textures;
    luisa::vector<Shape::Handle> _instances;
    luisa::vector<InstancedTransform> _dynamic_transforms;
    Buffer<Shape::Handle> _instance_buffer;
    luisa::vector<luisa::unique_ptr<Camera::Instance>> _cameras;
    luisa::vector<luisa::unique_ptr<Filter::Instance>> _filters;
    luisa::vector<luisa::unique_ptr<Film::Instance>> _films;
    luisa::unique_ptr<Integrator::Instance> _integrator;
    luisa::unique_ptr<LightSampler::Instance> _light_sampler;
    luisa::unique_ptr<Sampler::Instance> _sampler;
    luisa::unique_ptr<Environment::Instance> _environment;
    uint _rgb2spec_index{0u};
    float _mean_time{0.0f};

private:
    void _build_geometry(CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes, float init_time, AccelBuildHint hint) noexcept;
    void _process_shape(
        CommandBuffer &command_buffer, const Shape *shape,
        luisa::optional<bool> overridden_two_sided = luisa::nullopt,
        const Surface *overridden_surface = nullptr, const Light *overridden_light = nullptr) noexcept;
    [[nodiscard]] uint _process_surface(CommandBuffer &command_buffer, uint instance_id, const Shape *shape, const Surface *material) noexcept;
    [[nodiscard]] uint _process_light(CommandBuffer &command_buffer, uint instance_id, const Shape *shape, const Light *light) noexcept;

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

    [[nodiscard]] const TextureHandle *encode_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept;

    template<typename T, typename... Args>
        requires std::is_base_of_v<Resource, T>
    [[nodiscard]] auto create(Args &&...args) noexcept -> T * {
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

    [[nodiscard]] auto &device() const noexcept { return _device; }

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
    [[nodiscard]] auto surfaces() const noexcept { return luisa::span{_surfaces}; }
    [[nodiscard]] auto lights() const noexcept { return luisa::span{_lights}; }
    [[nodiscard]] auto sampler() noexcept { return _sampler.get(); }
    [[nodiscard]] auto sampler() const noexcept { return _sampler.get(); }
    [[nodiscard]] auto environment() const noexcept { return _environment.get(); }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler.get(); }
    [[nodiscard]] auto mean_time() const noexcept { return _mean_time; }

    bool update_geometry(CommandBuffer &command_buffer, float time) noexcept;
    void render(Stream &stream) noexcept;

    template<typename T, typename I>
    [[nodiscard]] auto buffer(I &&i) const noexcept { return _bindless_array.buffer<T>(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex2d(I &&i) const noexcept { return _bindless_array.tex2d(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex3d(I &&i) const noexcept { return _bindless_array.tex3d(std::forward<I>(i)); }

    [[nodiscard]] Var<Hit> trace_closest(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] luisa::unique_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept;
    [[nodiscard]] std::pair<Var<Shape::Handle>, Var<float4x4>> instance(Expr<uint> index) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Var<Shape::Handle> &instance, Expr<uint> index) const noexcept;
    [[nodiscard]] std::tuple<Var<float3> /* position */, Var<float3> /* ng */, Var<float> /* area */>
    surface_point_geometry(const Var<Shape::Handle> &instance, const Var<float4x4> &shape_to_world,
                           const Var<Triangle> &triangle, const Var<float3> &uvw) const noexcept;
    [[nodiscard]] std::tuple<Var<float3> /* ns */, Var<float3> /* tangent */, Var<float2> /* uv */>
    surface_point_attributes(const Var<Shape::Handle> &instance, const Var<float3x3> &shape_to_world_normal,
                             const Var<Triangle> &triangle, const Var<float3> &uvw) const noexcept;
    [[nodiscard]] auto intersect(const Var<Ray> &ray) const noexcept { return interaction(ray, trace_closest(ray)); }
    [[nodiscard]] auto intersect_any(const Var<Ray> &ray) const noexcept { return trace_any(ray); }

    void dynamic_dispatch_surface(Expr<uint> tag, const luisa::function<void(const Surface::Instance *)> &f) const noexcept;
    void dynamic_dispatch_light(Expr<uint> tag, const luisa::function<void(const Light::Instance *)> &f) const noexcept;

    [[nodiscard]] RGBAlbedoSpectrum srgb_albedo_spectrum(Expr<float3> rgb) const noexcept;
    [[nodiscard]] RGBUnboundSpectrum srgb_unbound_spectrum(Expr<float3> rgb) const noexcept;
    [[nodiscard]] RGBIlluminantSpectrum srgb_illuminant_spectrum(Expr<float3> rgb) const noexcept;
};

}// namespace luisa::render
