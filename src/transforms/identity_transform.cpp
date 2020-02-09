//
// Created by Mike Smith on 2020/2/10.
//

#include <core/mathematics.h>
#include "identity_transform.h"

namespace luisa {

float4x4 IdentityTransform::matrix(float time[[maybe_unused]]) const {
    return math::identity();
}

IdentityTransform::IdentityTransform(Device *device, const ParameterSet &parameter_set) noexcept
    : Transform{device, parameter_set} {}
    
}
