//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <render/plugin.h>
#include <render/parser.h>

namespace luisa::render {

using compute::Device;
using compute::KernelView;

class Filter : public Plugin {

protected:
    float _radius;

public:
    Filter(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _radius{params["radius"].parse_float_or_default(1.0f)} {}
    
    [[nodiscard]] float radius() const noexcept { return _radius; }
};

}
