//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/filter.h>
#include <core/film.h>

namespace luisa {

class MitchellNetravaliFilter : public SeparableFilter {

protected:
    float _b;
    float _c;
    
    [[nodiscard]] float _weight(float offset) const noexcept override;

public:
    MitchellNetravaliFilter(Device *device, const ParameterSet &parameters);
};

}
