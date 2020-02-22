//
// Created by Mike Smith on 2020/2/10.
//

#include "shape_instance.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("Instance", ShapeInstance)

void ShapeInstance::load(GeometryEncoder &encoder) {
    encoder.instantiate(this, _reference.get());
}

ShapeInstance::ShapeInstance(Device *device, const ParameterSet &parameter_set) : Shape{device, parameter_set} {
    _reference = parameter_set["reference"].parse<Shape>();
}

}
