//
// Created by Mike Smith on 2020/2/2.
//

#include <core/filter.h>

namespace luisa {

class MitchellNetravaliFilter : public SeparableFilter {

private:
    float _b;
    float _c;

protected:
    [[nodiscard]] float _weight_1d(float offset) const noexcept override {
        auto x = std::min(std::abs(2.0f * offset / _radius), 2.0f);
        return (x > 1.0f ?
                (-_b - 6.0f * _c) * x * x * x + (6.0f * _b + 30.0f * _c) * x * x + (-12.0f * _b - 48.0f * _c) * x + (8.0f * _b + 24.0f * _c) :
                (12.0f - 9.0f * _b - 6.0f * _c) * x * x * x + (-18.0f + 12.0f * _b + 6.0f * _c) * x * x + (6.0f - 2.0f * _b)) *
               (1.0f / 6.0f);
    }

public:
    MitchellNetravaliFilter(Device *device, const ParameterSet &parameters)
        : SeparableFilter{device, parameters},
          _b{parameters["b"].parse_float_or_default(1.0f / 3.0f)},
          _c{parameters["c"].parse_float_or_default(1.0f / 3.0f)} {}
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::MitchellNetravaliFilter)
