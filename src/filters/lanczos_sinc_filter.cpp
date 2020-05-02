//
// Created by Mike Smith on 2020/5/2.
//

#include <core/filter.h>

namespace luisa {

class LanczosSincFilter : public SeparableFilter {

private:
    float _tau;

protected:
    [[nodiscard]] float _weight_1d(float offset) const noexcept override {
        constexpr auto sinc = [](float x) noexcept {
            x = std::abs(x);
            return x < 1e-5f ? 1.0f : std::sin(math::PI * x) / (math::PI * x);
        };
        auto x = std::abs(offset);
        return x > _radius ? 0.0f : sinc(x) * sinc(x / _tau);
    }

public:
    LanczosSincFilter(Device *device, const ParameterSet &params)
        : SeparableFilter{device, params},
          _tau{params["tau"].parse_float_or_default(3.0f)} {}
};

LUISA_REGISTER_NODE_CREATOR("LanczosSinc", LanczosSincFilter)

}