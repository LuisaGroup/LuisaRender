//
// Created by Mike Smith on 2020/2/2.
//

#include "mitchell_netravali_filter.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("MitchellNetravali", MitchellNetravaliFilter)

MitchellNetravaliFilter::MitchellNetravaliFilter(Device *device, const ParameterSet &parameters)
    : SeparableFilter{device, parameters},
      _b{parameters["b"].parse_float_or_default(1.0f / 3.0f)},
      _c{parameters["c"].parse_float_or_default(1.0f / 3.0f)} {}

float MitchellNetravaliFilter::_weight(float offset) const noexcept {
    
    auto x = std::abs(2.0f * offset);
    return (x > 1.0f ?
            (-_b - 6.0f * _c) * x * x * x + (6.0f * _b + 30.0f * _c) * x * x + (-12.0f * _b - 48.0f * _c) * x + (8.0f * _b + 24.0f * _c) :
            (12.0f - 9.0f * _b - 6.0f * _c) * x * x * x + (-18.0f + 12.0f * _b + 6.0f * _c) * x * x + (6.0f - 2.0f * _b)) *
           (1.0f / 6.0f);
}
    
}
