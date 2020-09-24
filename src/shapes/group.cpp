//
// Created by Mike Smith on 2020/9/11.
//

#include <render/shape.h>

namespace luisa::render::shape {

struct Group : public Shape {
    Group(Device *device, const ParameterSet &params) : Shape{device, params} {
        _children = params["children"].parse_reference_list<Shape>();
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::shape::Group)
