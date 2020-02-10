//
// Created by Mike Smith on 2020/2/10.
//

#include "shape_instance.h"

namespace luisa {

void ShapeInstance::load(GeometryEncoder &encoder) {
    LUISA_ERROR_IF(_reference->is_instance(), "cannot make shape instance from instance");
    LUISA_ERROR_IF_NOT(_reference->transform().is_static(), "cannot make shape instance from shapes with non-static transforms");
    if (!_reference->loaded()) {
        _reference->load(encoder);
    }
    _entity_index = encoder.instantiate(_reference->entity_index());
}

ShapeInstance::ShapeInstance(Device *device, const ParameterSet &parameter_set) : Shape{device, parameter_set} {
    _reference = parameter_set["reference"].parse<Shape>();
}

}
