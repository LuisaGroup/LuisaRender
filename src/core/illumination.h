//
// Created by Mike Smith on 2020/2/12.
//

#pragma once

#include "data_types.h"

namespace luisa::illumination {

LUISA_DEVICE_CALLABLE inline void sample_lights() {}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "device.h"
#include "light.h"

namespace luisa {

class Illumination : Noncopyable {

private:
    Device *_device;
    std::vector<std::shared_ptr<Light>> _lights;

public:
    Illumination(Device *device, std::vector<std::shared_ptr<Light>> lights)
        : _device{device}, _lights{std::move(lights)} {}
    
    [[nodiscard]] static std::unique_ptr<Illumination> create(Device *device, std::vector<std::shared_ptr<Light>> lights) {
        return std::make_unique<Illumination>(device, std::move(lights));
    }
    
};

}

#endif