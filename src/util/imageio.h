//
// Created by Mike Smith on 2022/1/15.
//

#pragma once

#include <filesystem>

#include <core/stl.h>
#include <runtime/pixel.h>

namespace luisa::render {

// TODO: texture cache
class LoadedImage {

public:
    using storage_type = compute::PixelStorage;
    using deleter_type = void(*)(void *);

private:
    void *_pixels;
    uint2 _resolution;
    storage_type _storage;
    luisa::function<void(void *)> _deleter;

private:
    void _destroy() noexcept {
        if (_pixels != nullptr) {
            _deleter(_pixels);
        }
    }
    LoadedImage(void *pixels, storage_type storage, uint2 resolution, deleter_type deleter) noexcept
        : _pixels{pixels}, _resolution{resolution}, _storage{storage}, _deleter{deleter} {}
    [[nodiscard]] static LoadedImage _load_float(const std::filesystem::path &path, storage_type storage) noexcept;
    [[nodiscard]] static LoadedImage _load_byte(const std::filesystem::path &path, storage_type storage) noexcept;

public:
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
    [[nodiscard]] auto pixel_storage() const noexcept { return _storage; }
    [[nodiscard]] auto channels() const noexcept { return compute::pixel_storage_channel_count(_storage); }
    [[nodiscard]] auto pixel_size_bytes() const noexcept { return compute::pixel_storage_size(_storage); }
    [[nodiscard]] auto size_bytes() const noexcept { return _resolution.x * _resolution.y * pixel_size_bytes(); }

    [[nodiscard]] static LoadedImage load(const std::filesystem::path &path, storage_type storage) noexcept;
};

}
