//
// Created by Mike Smith on 2020/9/14.
//

#pragma once

#include <render/plugin.h>
#include <render/parser.h>

namespace luisa::render {

class Task : public Plugin {

public:
    Task(Device *device, const ParameterSet &params)
        : Plugin{device, params} {}

};

}
