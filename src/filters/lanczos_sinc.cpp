//
// Created by Mike Smith on 2022/1/16.
//

#include <base/filter.h>

namespace luisa::render {

class LanczosSincFilter final : public Filter {

private:
    float _tau;

public:
    LanczosSincFilter(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Filter{scene, desc}, _tau{desc->property_float_or_default("tau", 3.0f)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] float evaluate(float x) const noexcept override {
        x = x / radius();
        static constexpr auto sin_x_over_x = [](auto x) noexcept {
            return 1.0f + x * x == 1.0f ? 1.0f : std::sin(x) / x;
        };
        static constexpr auto sinc = [](auto x) noexcept {
            return sin_x_over_x(pi * x);
        };
        if (std::abs(x) > 1.0f) [[unlikely]] { return 0.0f; }
        return sinc(x) * sinc(x / _tau);
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LanczosSincFilter)
