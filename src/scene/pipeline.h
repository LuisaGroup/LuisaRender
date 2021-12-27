//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <optional>
#include <luisa-compute.h>

#include <scene/shape.h>
#include <scene/light.h>
#include <scene/material.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelBuildHint;
using compute::BindlessArray;
using compute::Buffer;
using compute::BufferView;
using compute::Device;
using compute::Image;
using compute::Mesh;
using compute::PixelStorage;
using compute::Resource;
using compute::Volume;
using compute::Callable;
using compute::Triangle;
using compute::BindlessBuffer;
using compute::BindlessTexture2D;
using compute::BindlessTexture3D;

class Scene;

class Pipeline {

public:
    template<typename T, size_t buffer_id_shift, size_t buffer_element_alignment>
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
        [[nodiscard]] std::pair<BufferView<T>, uint/* buffer id and offset */> allocate(size_t n) noexcept;
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
        uint resource_id;
        uint triangle_count;
        uint position_buffer_id_and_offset;
        uint attribute_buffer_id_and_offset;
        uint triangle_buffer_id_and_offset;
        uint area_cdf_buffer_id_and_offset;
    };

    struct MaterialInterface {
        uint tag;
        luisa::function<Material::Sample(/* TODO */)> sample;
        luisa::function<Material::Evaluation(/* TODO */)> evaluate;
    };

    struct LightInterface {
        uint tag;
        luisa::function<Light::Sample(/* TODO */)> sample;
        luisa::function<Light::Evaluation(/* TODO */)> evaluate;
    };

private:
    Device &_device;
    luisa::vector<ResourceHandle> _resources;
    BindlessArray _bindless_array;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    luisa::unordered_map<luisa::string/* impl type */, MaterialInterface> _material_interfaces;
    luisa::unordered_map<luisa::string/* impl type */, LightInterface> _light_interfaces;
    luisa::unordered_map<const Material *, uint/* buffer id and tag */> _materials;
    luisa::unordered_map<const Light *, uint/* buffer id and tag */> _lights;
    BufferArena<float3, Instance::position_buffer_id_shift, Instance::position_buffer_element_alignment> _position_buffer_arena;
    BufferArena<VertexAttribute, Instance::attribute_buffer_id_shift, Instance::attribute_buffer_element_alignment> _attribute_buffer_arena;
    BufferArena<Triangle, Instance::triangle_buffer_id_shift, Instance::triangle_buffer_element_alignment> _triangle_buffer_arena;
    BufferArena<float, Instance::area_cdf_buffer_id_shift, Instance::area_cdf_buffer_element_alignment> _area_cdf_buffer_arena;
    Accel _accel;
    luisa::vector<Instance> _instances;
    Buffer<Instance> _instance_buffer;
    luisa::unique_ptr<TransformTree> _transform_tree;

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
};

}// namespace luisa::render
