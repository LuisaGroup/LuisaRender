//
// Created by Mike Smith on 2020/2/9.
//

#include <core/mathematics.h>
#include "trs_transform.h"

namespace luisa {

TRSTransform::TRSTransform(Device *device, const ParameterSet &parameter_set)
    : Transform{device, parameter_set} {
    
    auto translation = parameter_set["translation"].parse_float3_or_default(make_float4());
    auto rotation = parameter_set["rotation"].parse_float4_or_default(make_float4(0.0f, 1.0f, 0.0f, 0.0f));
    auto scaling = parameter_set["scaling"].parse_float3_or_default(make_float3(1.0f));
    
    _matrix = math::translation(translation) * math::rotation(make_float3(rotation), rotation.w) * math::scaling(scaling);
}

}
