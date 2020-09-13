//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <render/plugin.h>
#include <render/parser.h>

namespace luisa::render {

using compute::TextureView;
using compute::BufferView;
using compute::Dispatcher;

class Film : public Plugin {

private:
    uint2 _resolution;

protected:
    virtual void _clear(Dispatcher &dispatch) = 0;

public:
    Film(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _resolution{params["resolution"].parse_uint2_or_default(make_uint2(1280u, 720u))} {}
    
    [[nodiscard]] uint2 resolution() const noexcept { return _resolution; }
    
    // void
};

}
