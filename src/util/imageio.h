//
// Created by Mike Smith on 2022/1/15.
//

#pragma once

#include <filesystem>

#include <core/stl.h>
#include <util/half.h>
#include <runtime/rhi/pixel.h>

namespace luisa::render {

// TODO: texture cache
class LoadedImage {

public:
    using storage_type = compute::PixelStorage;

private:
    void *_pixels{nullptr};
    uint2 _resolution;
    storage_type _storage{};
    luisa::function<void(void *)> _deleter;

private:
    void _destroy() noexcept;
    LoadedImage(void *pixels, storage_type storage, uint2 resolution, luisa::function<void(void *)> deleter) noexcept;
    [[nodiscard]] static LoadedImage _load_byte(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_half(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_short(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_float(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_int(const std::filesystem::path &path, storage_type storage) noexcept;

public:
    LoadedImage() noexcept = default;
    ~LoadedImage() noexcept;
    LoadedImage(LoadedImage &&another) noexcept;
    LoadedImage &operator=(LoadedImage &&rhs) noexcept;
    LoadedImage(const LoadedImage &) noexcept = delete;
    LoadedImage &operator=(const LoadedImage &) noexcept = delete;
    [[nodiscard]] auto size() const noexcept { return _resolution; }
    [[nodiscard]] void *pixels(uint level = 0u) noexcept { return _pixels; }
    [[nodiscard]] const void *pixels(uint level = 0u) const noexcept { return _pixels; }
    [[nodiscard]] auto pixel_storage() const noexcept { return _storage; }
    [[nodiscard]] auto channels() const noexcept { return compute::pixel_storage_channel_count(_storage); }
    [[nodiscard]] auto pixel_count() const noexcept { return _resolution.x * _resolution.y; }
    [[nodiscard]] explicit operator bool() const noexcept { return _pixels != nullptr; }
    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path) noexcept;
    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static storage_type parse_storage(const std::filesystem::path &path) noexcept;
    [[nodiscard]] static LoadedImage create(uint2 resolution, storage_type storage) noexcept;
};

void save_image(std::filesystem::path path, const float *pixels,
                uint2 resolution, uint components = 4) noexcept;

void save_image(std::filesystem::path path, const uint8_t *pixels,
                uint2 resolution, uint components = 4) noexcept;

}// namespace luisa::render
