//
// Created by Mike Smith on 2022/1/15.
//

#pragma once

#include <filesystem>

#include <core/stl.h>
#include <core/basic_types.h>

namespace luisa::render {

// TODO: texture cache

template<typename T>
class LoadedImage {

private:
    T *_pixels;
    uint2 _resolution;
    uint _channels;
    void(*_deleter)(T *);

private:
    void _destroy() noexcept {
        if (_pixels != nullptr) { _deleter(_pixels); }
    }

public:
    LoadedImage(T *pixels, uint2 resolution, uint num_channels, void (*deleter)(T *)) noexcept
        : _pixels{pixels}, _resolution{resolution}, _channels{num_channels}, _deleter{deleter} {}
    ~LoadedImage() noexcept { _destroy(); }
    LoadedImage(LoadedImage &&another) noexcept
        : _pixels{another._pixels},
          _resolution{another._resolution},
          _channels{another._channels},
          _deleter{another._deleter} { another._pixels = nullptr; }
    LoadedImage &operator=(LoadedImage &&rhs) noexcept {
        if (&rhs != this) [[likely]] {
            _destroy();
            _pixels = rhs._pixels;
            _resolution = rhs._resolution;
            _channels = rhs._channels;
            _deleter = rhs._deleter;
            rhs._pixels = nullptr;
        }
    }
    LoadedImage(const LoadedImage &) noexcept = delete;
    LoadedImage &operator=(const LoadedImage &) noexcept = delete;
    [[nodiscard]] auto pixels() const noexcept { return _pixels; }
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] auto num_channels() const noexcept { return _channels; }
};

[[nodiscard]] LoadedImage<float> load_hdr_image(const std::filesystem::path &path, uint expected_channels = 0u) noexcept;
[[nodiscard]] LoadedImage<uint8_t> load_ldr_image(const std::filesystem::path &path, uint expected_channels = 0u) noexcept;

}
