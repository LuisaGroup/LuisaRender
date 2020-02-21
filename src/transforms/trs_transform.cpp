//
// Created by Mike Smith on 2020/2/9.
//

#include <core/mathematics.h>
#include "trs_transform.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("TRS", TRSTransform)

TRSTransform::TRSTransform(Device *device, const ParameterSet &parameter_set)
    : Transform{device, parameter_set},
      _t{parameter_set["translation"].parse_float3_or_default(make_float4())},
      _r{parameter_set["rotation"].parse_float4_or_default(make_float4(0.0f, 1.0f, 0.0f, 0.0f))},
      _s{parameter_set["scaling"].parse_float3_or_default(make_float3(parameter_set["scaling"].parse_float_or_default(1.0f)))} {
    
    _r.w = math::radians(_r.w);
    _matrix = math::translation(_t) * math::rotation(make_float3(_r), _r.w) * math::scaling(_s);
}

}
