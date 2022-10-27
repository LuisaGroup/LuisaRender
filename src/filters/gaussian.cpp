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
        : Filter{scene, desc}, _sigma{desc->property_float_or_default("sigma", 0.f)} {
        if (_sigma <= 0.f) {// invalid sigma, compute from radius
            auto r = max(radius().x, radius().y);
            // G = 1.f / (sqrt(2.f * pi) * sigma) * exp(-r * r / (2.f * sigma * sigma)).
            // Ignoring the normalization factor, we have
            // F(r) = exp(-r * r / (2.f * sigma * sigma)),
            // where k is a constant. Let F(radius) = eps, we have
            // sigma = sqrt(radius * radius / (-2 * log(eps)))
            //       = radius / sqrt(-2.f * log(eps)).
            // We choose eps = 1e-2, so that approximately, sigma = radius / 3.
            _sigma = r / 3.f;
        }
    }
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
