//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/buffer_arena.h>
#include <runtime/image.h>
#include <runtime/bindless_array.h>
#include <runtime/rtx/mesh.h>
#include <runtime/rtx/accel.h>

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
#include <base/geometry.h>
#include <base/medium.h>
#include <base/phase_function.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelOption;
using compute::BindlessArray;
using compute::BindlessBuffer;
using compute::BindlessTexture2D;
using compute::BindlessTexture3D;
using compute::Buffer;
using compute::BufferArena;
using compute::BufferView;
using compute::Callable;
using compute::Device;
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
    static constexpr auto transform_matrix_buffer_size = 65536u;
    static constexpr auto constant_buffer_size = 256u * 1024u;
    using ResourceHandle = luisa::unique_ptr<Resource>;

private:
    Device &_device;
    BindlessArray _bindless_array;
    luisa::unique_ptr<BufferArena> _general_buffer_arena;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    luisa::vector<ResourceHandle> _resources;
    Buffer<float4> _constant_buffer;
    size_t _constant_count{0u};
    Polymorphic<Surface::Instance> _surfaces;
    Polymorphic<Light::Instance> _lights;
    Polymorphic<Medium::Instance> _media;
    luisa::unordered_map<const Surface *, uint> _surface_tags;
    luisa::unordered_map<const Light *, uint> _light_tags;
    luisa::unordered_map<const Medium *, uint> _medium_tags;
    luisa::unordered_map<const Texture *, luisa::unique_ptr<Texture::Instance>> _textures;
    luisa::unordered_map<const Filter *, luisa::unique_ptr<Filter::Instance>> _filters;
    luisa::unordered_map<const PhaseFunction *, luisa::unique_ptr<PhaseFunction::Instance>> _phasefunctions;
    luisa::vector<luisa::unique_ptr<Camera::Instance>> _cameras;
    luisa::unique_ptr<Spectrum::Instance> _spectrum;
    luisa::unique_ptr<Integrator::Instance> _integrator;
    luisa::unique_ptr<Environment::Instance> _environment;
    uint _environment_medium_tag{Medium::INVALID_TAG};
    luisa::unique_ptr<Differentiation> _differentiation;
    luisa::unique_ptr<Geometry> _geometry;
    // registered transforms
    luisa::unordered_map<const Transform *, uint> _transform_to_id;
    luisa::vector<const Transform *> _transforms;
    luisa::vector<float4x4> _transform_matrices;
    Buffer<float4x4> _transform_matrix_buffer;
    luisa::unordered_map<luisa::string, uint> _named_ids;
    // other things
    luisa::unique_ptr<Printer> _printer;
    float _initial_time{};
    bool _any_dynamic_transforms{false};

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
        _bindless_array.emplace_on_update(buffer_id, buffer);
        return static_cast<uint>(buffer_id);
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Buffer<T> &buffer) noexcept {
        return register_bindless(buffer.view());
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Image<T> &image, TextureSampler sampler) noexcept {
        auto tex2d_id = _bindless_tex2d_count++;
        _bindless_array.emplace_on_update(tex2d_id, image, sampler);
        return static_cast<uint>(tex2d_id);
    }

    template<typename T>
    [[nodiscard]] auto register_bindless(const Volume<T> &volume, TextureSampler sampler) noexcept {
        auto tex3d_id = _bindless_tex3d_count++;
        _bindless_array.emplace_on_update(tex3d_id, volume, sampler);
        return static_cast<uint>(tex3d_id);
    }

    void register_transform(const Transform *transform) noexcept;

    [[nodiscard]] uint register_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept;
    [[nodiscard]] uint register_light(CommandBuffer &command_buffer, const Light *light) noexcept;
    [[nodiscard]] uint register_medium(CommandBuffer &command_buffer, const Medium *medium) noexcept;

    template<typename Create>
    uint register_named_id(luisa::string_view identifier, Create &&create_id) noexcept {
        if (auto it = _named_ids.find(identifier); it != _named_ids.end()) {
            return it->second;
        }
        auto new_id = std::invoke(std::forward<Create>(create_id));
        _named_ids.emplace(identifier, new_id);
        return new_id;
    }

    template<typename T, typename... Args>
        requires std::is_base_of_v<Resource, T>
    [[nodiscard]] auto create(Args &&...args) noexcept -> T * {
        auto resource = luisa::make_unique<T>(_device.create<T>(std::forward<Args>(args)...));
        auto p = resource.get();
        _resources.emplace_back(std::move(resource));
        return p;
    }

    template<uint dim, typename Def>
    void register_shader(luisa::string_view name, Def &&def) noexcept {
        static_assert(dim == 1u || dim == 2u || dim == 3u);
        register_named_id(name, [&] {
            auto shader = _device.compile<dim>(std::forward<Def>(def));
            auto resource = luisa::make_unique<decltype(shader)>(std::move(shader));
            auto index = static_cast<uint>(_resources.size());
            _resources.emplace_back(std::move(resource));
            return index;
        });
    }

    template<typename T>
    [[nodiscard]] BufferView<T> arena_buffer(size_t n) noexcept {
        return create<Buffer<T>>(n)->view();
        // FIXME: buffer arena is broken
        return _general_buffer_arena->allocate<T>(
            std::max(n, static_cast<size_t>(1u)));
    }

    template<typename T>
    [[nodiscard]] std::pair<BufferView<T>, uint /* bindless id */> bindless_arena_buffer(size_t n) noexcept {
        auto view = arena_buffer<T>(n);
        auto buffer_id = register_bindless(view);
        return std::make_pair(view, buffer_id);
    }

    [[nodiscard]] std::pair<BufferView<float4>, uint> allocate_constant_slot() noexcept;

public:
    [[nodiscard]] auto &device() const noexcept { return _device; }
    [[nodiscard]] static luisa::unique_ptr<Pipeline> create(
        Device &device, Stream &stream, const Scene &scene) noexcept;
    [[nodiscard]] Differentiation *differentiation() noexcept;
    [[nodiscard]] const Differentiation *differentiation() const noexcept;
    [[nodiscard]] auto &bindless_array() noexcept { return _bindless_array; }
    [[nodiscard]] auto &bindless_array() const noexcept { return _bindless_array; }
    [[nodiscard]] auto camera_count() const noexcept { return _cameras.size(); }
    [[nodiscard]] auto camera(size_t i) noexcept { return _cameras[i].get(); }
    [[nodiscard]] auto camera(size_t i) const noexcept { return _cameras[i].get(); }
    [[nodiscard]] auto &surfaces() const noexcept { return _surfaces; }
    [[nodiscard]] auto &lights() const noexcept { return _lights; }
    [[nodiscard]] auto &media() const noexcept { return _media; }
    [[nodiscard]] auto environment() const noexcept { return _environment.get(); }
    [[nodiscard]] auto environment_medium_tag() const noexcept { return _environment_medium_tag; }
    [[nodiscard]] auto integrator() const noexcept { return _integrator.get(); }
    [[nodiscard]] auto spectrum() const noexcept { return _spectrum.get(); }
    [[nodiscard]] auto geometry() const noexcept { return _geometry.get(); }
    [[nodiscard]] auto has_lighting() const noexcept { return !_lights.empty() || _environment != nullptr; }
    [[nodiscard]] const Texture::Instance *build_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept;
    [[nodiscard]] const Filter::Instance *build_filter(CommandBuffer &command_buffer, const Filter *filter) noexcept;
    [[nodiscard]] const PhaseFunction::Instance *build_phasefunction(CommandBuffer &command_buffer, const PhaseFunction *phasefunction) noexcept;
    bool update(CommandBuffer &command_buffer, float time) noexcept;
    void render(Stream &stream) noexcept;
    [[nodiscard]] auto &printer() noexcept { return *_printer; }
    [[nodiscard]] auto &printer() const noexcept { return *_printer; }
    [[nodiscard]] uint named_id(luisa::string_view name) const noexcept;
    template<typename T, typename I>
    [[nodiscard]] auto buffer(I &&i) const noexcept { return _bindless_array->buffer<T>(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex2d(I &&i) const noexcept { return _bindless_array->tex2d(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex3d(I &&i) const noexcept { return _bindless_array->tex3d(std::forward<I>(i)); }
    template<typename T>
    [[nodiscard]] auto named_buffer(luisa::string_view name) const noexcept {
        return _bindless_array->buffer<T>(named_id(name));
    }
    [[nodiscard]] auto named_tex2d(luisa::string_view name) const noexcept {
        return _bindless_array->tex2d(named_id(name));
    }
    [[nodiscard]] auto named_tex3d(luisa::string_view name) const noexcept {
        return _bindless_array->tex3d(named_id(name));
    }
    [[nodiscard]] Float4x4 transform(const Transform *transform) const noexcept;

    [[nodiscard]] Float4 constant(Expr<uint> index) const noexcept;

    template<uint dim, typename... Args, typename... CallArgs>
    [[nodiscard]] auto shader(luisa::string_view name, CallArgs &&...call_args) const noexcept {
        auto shader = dynamic_cast<const Shader<dim, Args...> *>(
            _resources[named_id(name)].get());
        return (*shader)(std::forward<CallArgs>(call_args)...);
    }
};

}// namespace luisa::render
