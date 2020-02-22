//
// Created by Mike Smith on 2020/2/10.
//

#include "shape_replica.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("Replica", ShapeReplica)

void ShapeReplica::load(GeometryEncoder &encoder) {
    encoder.replicate(this, _reference.get());
}

ShapeReplica::ShapeReplica(Device *device, const ParameterSet &parameter_set)
    : Shape{device, parameter_set} { _reference = parameter_set["reference"].parse<Shape>(); }
    
}
