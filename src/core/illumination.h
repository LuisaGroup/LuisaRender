//
// Created by Mike Smith on 2020/2/21.
//

#pragma once

#include "data_types.h"

namespace luisa::illumination {

class Info {

private:
    uint8_t _tag;
    uint8_t _index_hi;
    uint16_t _index_lo;

public:
    constexpr Info(uint tag, uint index) noexcept
        : _tag{static_cast<uint8_t>(tag)}, _index_hi{static_cast<uint8_t>(index >> 24u)}, _index_lo{static_cast<uint16_t>(index)} {}
    
    [[nodiscard]] constexpr auto tag() const noexcept { return static_cast<uint>(_tag); }
    [[nodiscard]] constexpr auto index() const noexcept { return (static_cast<uint>(_index_hi) << 24u) | static_cast<uint>(_index_lo); }
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <vector>

#include "light.h"
#include "geometry.h"
#include "kernel.h"

namespace luisa {

class Illumination {

private:
    Device *_device;
    Geometry *_geometry;
    std::vector<std::shared_ptr<Light>> _lights;
    
    std::unique_ptr<Buffer<illumination::Info>> _info_buffer;
    std::vector<std::unique_ptr<Kernel>> _light_sampling_kernels;
    std::vector<std::unique_ptr<TypelessBuffer>> _light_data_buffers;

public:
    Illumination(Device *device, std::vector<std::shared_ptr<Light>> lights, Geometry *geometry);
    
};

}

#endif