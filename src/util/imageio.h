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
    uint2 _resolution;
    storage_type _storage{};
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
          _resolution{another._resolution},
          _storage{another._storage},
          _deleter{another._deleter} { another._pixels = nullptr; }
    LoadedImage &operator=(LoadedImage &&rhs) noexcept {
        if (&rhs != this) [[likely]] {
            _destroy();
            _pixels = rhs._pixels;
            _resolution = rhs._resolution;
            _storage = rhs._storage;
            _deleter = rhs._deleter;
            rhs._pixels = nullptr;
        }
        return *this;
    }
    LoadedImage(const LoadedImage &) noexcept = delete;
    LoadedImage &operator=(const LoadedImage &) noexcept = delete;
    [[nodiscard]] auto size() const noexcept { return _resolution; }
    [[nodiscard]] auto pixels() const noexcept { return _pixels; }
    void set_pixel_storage(storage_type s) noexcept { _storage = s; }
    [[nodiscard]] auto pixel_storage() const noexcept { return _storage; }
    [[nodiscard]] auto channels() const noexcept { return compute::pixel_storage_channel_count(_storage); }
    [[nodiscard]] auto pixel_size_bytes() const noexcept { return compute::pixel_storage_size(_storage); }
    [[nodiscard]] auto pixel_count() const noexcept { return _resolution.x * _resolution.y; }
    [[nodiscard]] auto size_bytes() const noexcept { return pixel_count() * pixel_size_bytes(); }
    [[nodiscard]] explicit operator bool() const noexcept { return _pixels != nullptr; }
    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path) noexcept;
    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static storage_type parse_storage(const std::filesystem::path &path) noexcept;
    [[nodiscard]] static LoadedImage create(uint2 resolution, storage_type storage) noexcept;
    [[nodiscard]] float4 read(uint2 p) const noexcept;
    void write(uint2 p, float4 v) noexcept;
};

class TiledMipmap {

private:
    uint2 _size;
    luisa::vector<std::byte> _data;

public:
};

void save_image(std::filesystem::path path, const float *pixels,
                uint2 resolution, uint components = 4) noexcept;

}// namespace luisa::render
