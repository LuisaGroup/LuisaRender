//
// Created by Mike Smith on 2020/9/14.
//

#pragma once

#include <render/plugin.h>
#include <render/parser.h>

namespace luisa::render {

class Integrator : public Plugin {

private:


public:
    Integrator(Device *device, const ParameterSet &params) noexcept
        : Plugin{device, params} {}

};

}
