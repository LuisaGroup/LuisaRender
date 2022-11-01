//
// Created by Mike Smith on 2022/1/15.
//

#pragma once

#include <filesystem>

#include <core/stl.h>
#include <util/half.h>
#include <runtime/pixel.h>

namespace luisa::render {

// TODO: texture cache
class LoadedImage {

public:
    using storage_type = compute::PixelStorage;

private:
    void *_pixels{nullptr};
    void *_mipmap_pixels{nullptr};
    uint2 _resolution;
    storage_type _storage{};
    uint _mipmap_levels{0u};
    luisa::function<void(void *)> _deleter;

private:
    void _destroy() noexcept;
    LoadedImage(void *pixels, storage_type storage, uint2 resolution, luisa::function<void(void *)> deleter) noexcept
        : _pixels{pixels}, _resolution{resolution}, _storage{storage}, _deleter{std::move(deleter)} {}
    [[nodiscard]] static LoadedImage _load_byte(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_half(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_short(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_float(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_int(const std::filesystem::path &path, storage_type storage) noexcept;

public:
    LoadedImage() noexcept = default;
    ~LoadedImage() noexcept { _destroy(); }
    LoadedImage(LoadedImage &&another) noexcept
        : _pixels{another._pixels},
          _mipmap_pixels{another._mipmap_pixels},
          _resolution{another._resolution},
          _storage{another._storage},
          _mipmap_levels{another._mipmap_levels},
          _deleter{std::move(another._deleter)} {
        another._pixels = nullptr;
        another._mipmap_pixels = nullptr;
    }
    LoadedImage &operator=(LoadedImage &&rhs) noexcept {
        if (&rhs != this) [[likely]] {
            _destroy();
            _pixels = rhs._pixels;
            _mipmap_pixels = rhs._mipmap_pixels;
            _resolution = rhs._resolution;
            _storage = rhs._storage;
            _mipmap_levels = rhs._mipmap_levels;
            _deleter = std::move(rhs._deleter);
            rhs._pixels = nullptr;
            rhs._mipmap_pixels = nullptr;
        }
        return *this;
    }
    LoadedImage(const LoadedImage &) noexcept = delete;
    LoadedImage &operator=(const LoadedImage &) noexcept = delete;
    [[nodiscard]] auto size() const noexcept { return _resolution; }
    [[nodiscard]] void *pixels(uint level = 0u) noexcept;
    [[nodiscard]] const void *pixels(uint level = 0u) const noexcept;
    [[nodiscard]] auto pixel_storage() const noexcept { return _storage; }
    [[nodiscard]] auto channels() const noexcept { return compute::pixel_storage_channel_count(_storage); }
    [[nodiscard]] auto pixel_count() const noexcept { return _resolution.x * _resolution.y; }
    [[nodiscard]] explicit operator bool() const noexcept { return _pixels != nullptr; }
    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path) noexcept;
    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static storage_type parse_storage(const std::filesystem::path &path) noexcept;
    [[nodiscard]] static LoadedImage create(uint2 resolution, storage_type storage) noexcept;
    void generate_mipmaps(uint levels = 0u) noexcept;
    [[nodiscard]] auto mipmap_levels() const noexcept { return std::max(_mipmap_levels, 1u); }
};

void save_image(std::filesystem::path path, const float *pixels,
                uint2 resolution, uint components = 4) noexcept;

}// namespace luisa::render
