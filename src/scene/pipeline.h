//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <optional>
#include <luisa-compute.h>

#include <scene/shape.h>

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

class Scene;
class Shape;

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

    private:
        void _create_buffer() noexcept;

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

private:
    Device &_device;
    luisa::vector<ResourceHandle> _resources;
    BindlessArray _bindless_array;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    Accel _accel;
    luisa::unordered_map<const Shape *, MeshData> _meshes;
    luisa::unordered_map<const Material *, uint/* buffer id and tag */> _materials;
    luisa::unordered_map<const Light *, uint/* buffer id and tag */> _lights;
    BufferArena<float3, Instance::position_buffer_id_shift, Instance::position_buffer_element_alignment> _position_buffer_arena;
    BufferArena<VertexAttribute, Instance::attribute_buffer_id_shift, Instance::attribute_buffer_element_alignment> _attribute_buffer_arena;
    BufferArena<Triangle, Instance::triangle_buffer_id_shift, Instance::triangle_buffer_element_alignment> _triangle_buffer_arena;
    BufferArena<float, Instance::area_cdf_buffer_id_shift, Instance::area_cdf_buffer_element_alignment> _area_cdf_buffer_arena;
    BufferArena<float4x4, Instance::transform_buffer_id_shift, 1u> _transform_buffer_arena;
    BufferArena<Instance, Instance::instance_buffer_id_shift, 1u> _instance_buffer_arena;

public:
    // for internal use only; use Pipeline::create() instead
    explicit Pipeline(Device &device) noexcept;
    Pipeline(Pipeline &&) noexcept = delete;
    Pipeline(const Pipeline &) noexcept = delete;
    Pipeline &operator=(Pipeline &&) noexcept = delete;
    Pipeline &operator=(const Pipeline &) noexcept = delete;
    ~Pipeline() noexcept;

private:
    template<typename T>
    [[nodiscard]] auto _emplace_back_bindless_buffer(BufferView<T> buffer) noexcept {
        auto buffer_id = _bindless_buffer_count++;
        _bindless_array.emplace(buffer_id, buffer);
        return static_cast<uint>(buffer_id);
    }

public:
    template<typename T>
    [[nodiscard]] std::pair<BufferView<T>, uint/* id */> create_bindless_buffer(size_t size) noexcept {

    }

    // low-level interfaces, for internal resources of scene nodes
//    template<typename T>
//    [[nodiscard]] Buffer<T> &create_buffer(size_t size) noexcept {
//        auto buffer = luisa::make_unique<Buffer<T>>(_device.create_buffer<T>(size));
//        auto p_buffer = buffer.get();
//        _resources.emplace_back(std::move(buffer));
//        return *p_buffer;
//    }
//
//    template<typename T>
//    [[nodiscard]] Image<T> &create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
//        auto image = luisa::make_unique<Image<T>>(_device.create_image<T>(pixel, size, mip_levels));
//        auto p_image = image.get();
//        _resources.emplace_back(std::move(image));
//        return *p_image;
//    }
//
//    template<typename T>
//    [[nodiscard]] Volume<T> &create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
//        auto volume = luisa::make_unique<Volume<T>>(_device.create_volume<T>(pixel, size, mip_levels));
//        auto p_volume = volume.get();
//        _resources.emplace_back(std::move(volume));
//        return *p_volume;
//    }
//
//    template<typename VBuffer, typename TBuffer>
//    [[nodiscard]] Mesh &create_mesh(
//        VBuffer &&vertices, TBuffer &&triangles,
//        AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept {
//        auto mesh = luisa::make_unique<Mesh>(_device.create_mesh(
//            std::forward<VBuffer>(vertices),
//            std::forward<TBuffer>(triangles)));
//        auto p_mesh = mesh.get();
//        _resources.emplace_back(std::move(mesh));
//        return *p_mesh;
//    }
//
//    [[nodiscard]] Accel &create_accel(AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept {
//        auto accel = luisa::make_unique<Accel>(_device.create_accel(hint));
//        auto p_accel = accel.get();
//        _resources.emplace_back(std::move(accel));
//        return *p_accel;
//    }
//
//    [[nodiscard]] BindlessArray &create_bindless_array(size_t capacity = 65536u) noexcept {
//        auto array = luisa::make_unique<BindlessArray>(_device.create_bindless_array(capacity));
//        auto p_array = array.get();
//        _resources.emplace_back(std::move(array));
//        return *p_array;
//    }
};

}// namespace luisa::render
