//
// Created by Mike Smith on 2020/5/2.
//

#include "transform.h"

namespace luisa {

IdentityTransform::IdentityTransform(Device *device) noexcept: Transform{device} {}

IdentityTransform::IdentityTransform(Device *device, const ParameterSet &parameter_set) noexcept
    : Transform{device, parameter_set} {}

LUISA_REGISTER_NODE_CREATOR("Identity", IdentityTransform)
    
}