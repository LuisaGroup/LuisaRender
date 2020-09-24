//
// Created by Mike Smith on 2020/9/11.
//

#include <render/shape.h>

namespace luisa::render::shape {

struct Instance : public Shape {
    Instance(Device *device, const ParameterSet &params) : Shape{device, params} {
        _children.emplace_back(params["reference"].parse<Shape>());
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::shape::Instance)
