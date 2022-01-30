//
// Created by Mike Smith on 2022/1/16.
//

#include <base/filter.h>

namespace luisa::render {

class GaussianFilter final : public Filter {

private:
    float _sigma;

public:
    GaussianFilter(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Filter{scene, desc}, _sigma{desc->property_float_or_default("sigma", 0.5f)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] float evaluate(float x) const noexcept override {
        auto G = [s = 2.0f * _sigma * _sigma](auto x) noexcept {
            return 1.0f / std::sqrt(pi * s) * std::exp(-x * x / s);
        };
        return G(x) - G(1.0f);
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GaussianFilter)
