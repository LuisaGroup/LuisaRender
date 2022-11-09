//
// Created by Mike Smith on 2022/1/16.
//

#include <base/filter.h>

namespace luisa::render {

class MitchellFilter final : public Filter {

private:
    float _b;
    float _c;

public:
    MitchellFilter(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Filter{scene, desc},
          _b{desc->property_float_or_default("b", 1.0f / 3.0f)},
          _c{desc->property_float_or_default("c", 1.0f / 3.0f)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] float evaluate(float x) const noexcept override {
        x = 2.f * std::abs(x / radius());
        if (x <= 1.0f) {
            return ((12.0f - 9.0f * _b - 6.0f * _c) * x * x * x +
                    (-18.0f + 12.0f * _b + 6.0f * _c) * x * x +
                    (6.0f - 2.0f * _b)) *
                   (1.f / 6.f);
        }
        if (x <= 2.0f) {
            return ((-_b - 6.0f * _c) * x * x * x +
                    (6.0f * _b + 30.0f * _c) * x * x +
                    (-12.0f * _b - 48.0f * _c) * x +
                    (8.0f * _b + 24.0f * _c)) *
                   (1.0f / 6.0f);
        }
        return 0.0f;
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MitchellFilter)
