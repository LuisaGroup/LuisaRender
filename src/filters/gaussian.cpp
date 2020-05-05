//
// Created by Mike Smith on 2020/5/2.
//

#include <core/filter.h>

namespace luisa {

class GaussianFilter : public SeparableFilter {

private:
    float _alpha;

protected:
    [[nodiscard]] float _weight_1d(float offset) const noexcept override {
        return std::max(0.0f, std::exp(-_alpha * offset * offset) - std::exp(-_alpha * _radius * _radius));
    }

public:
    GaussianFilter(Device *device, const ParameterSet &parameters)
        : SeparableFilter{device, parameters},
          _alpha{parameters["alpha"].parse_float_or_default(2.0f)} {}
};

LUISA_REGISTER_NODE_CREATOR("Gaussian", GaussianFilter)

}
