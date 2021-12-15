//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <optional>
#include <luisa-compute.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelBuildHint;
using compute::BindlessArray;
using compute::Buffer;
using compute::Device;
using compute::Image;
using compute::Mesh;
using compute::PixelStorage;
using compute::Resource;
using compute::Volume;
using compute::Callable;

class Scene;

class Pipeline {

public:
    using ResourceHandle = luisa::unique_ptr<Resource>;

private:
    Device &_device;
    luisa::vector<ResourceHandle> _resources;

public:
    // for internal use only; use Pipeline::create() instead
    explicit Pipeline(Device &device) noexcept;
    Pipeline(Pipeline &&) noexcept = delete;
    Pipeline(const Pipeline &) noexcept = delete;
    Pipeline &operator=(Pipeline &&) noexcept = delete;
    Pipeline &operator=(const Pipeline &) noexcept = delete;
    ~Pipeline() noexcept = default;

public:
    // low-level interfaces, for internal resources of scene nodes
    template<typename T>
    [[nodiscard]] Buffer<T> &create_buffer(size_t size) noexcept {
        auto buffer = luisa::make_unique<Buffer<T>>(_device.create_buffer<T>(size));
        auto p_buffer = buffer.get();
        _resources.emplace_back(std::move(buffer));
        return *p_buffer;
    }

    template<typename T>
    [[nodiscard]] Image<T> &create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
        auto image = luisa::make_unique<Image<T>>(_device.create_image<T>(pixel, size, mip_levels));
        auto p_image = image.get();
        _resources.emplace_back(std::move(image));
        return *p_image;
    }

    template<typename T>
    [[nodiscard]] Volume<T> &create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
        auto volume = luisa::make_unique<Volume<T>>(_device.create_volume<T>(pixel, size, mip_levels));
        auto p_volume = volume.get();
        _resources.emplace_back(std::move(volume));
        return *p_volume;
    }

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh &create_mesh(
        VBuffer &&vertices, TBuffer &&triangles,
        AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept {
        auto mesh = luisa::make_unique<Mesh>(_device.create_mesh(
            std::forward<VBuffer>(vertices),
            std::forward<TBuffer>(triangles)));
        auto p_mesh = mesh.get();
        _resources.emplace_back(std::move(mesh));
        return *p_mesh;
    }

    [[nodiscard]] Accel &create_accel(AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept {
        auto accel = luisa::make_unique<Accel>(_device.create_accel(hint));
        auto p_accel = accel.get();
        _resources.emplace_back(std::move(accel));
        return *p_accel;
    }

    [[nodiscard]] BindlessArray &create_bindless_array(size_t capacity = 65536u) noexcept {
        auto array = luisa::make_unique<BindlessArray>(_device.create_bindless_array(capacity));
        auto p_array = array.get();
        _resources.emplace_back(std::move(array));
        return *p_array;
    }
};

}// namespace luisa::render
