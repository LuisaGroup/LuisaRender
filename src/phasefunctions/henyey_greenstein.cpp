//
// Created by ChenXin on 2023/2/14.
//

#include <base/phase_function.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

// https://pbr-book.org/3ed-2018/Volume_Scattering/Phase_Functions#PhaseHG
class HenyeyGreenstein : public PhaseFunction {
private:
    float _g;

public:
    class HenyeyGreensteinInstance : public PhaseFunction::Instance {
    protected:
        friend class HenyeyGreenstein;

    public:
        [[nodiscard]] Float p(Expr<float3> wo, Expr<float3> wi) const override {
            auto g = node<HenyeyGreenstein>()->_g;
            auto cosTheta = dot(wo, wi);
            auto denom = 1.f + sqr(g) + 2.f * g * cosTheta;
            return inv_pi * 0.25f * (1.f - sqr(g)) / (denom * sqrt(max(0.f, denom)));
        }
        [[nodiscard]] PhaseFunctionSample sample_p(Expr<float3> wo, Expr<float2> u) const override {
            auto g = node<HenyeyGreenstein>()->_g;
            // sample cosTheta
            auto cosTheta = ite(
                std::abs(g) < 1e-3f,
                1.f - 2.f * u.x,
                //        1.f + g * g - sqr((1.f - g * g) / (1 - g + 2 * g * u.x)) / (2.f * g)
                -1.f / (2.f * g) * (1.f + sqr(g) - sqr((1.f - sqr(g)) / (1.f + g - 2.f * g * u.x))));

            // compute direction
            auto sinTheta = sqrt(max(0.f, 1.f - sqr(cosTheta)));
            auto phi = 2.f * pi * u.y;

            auto wi = make_float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
            auto p_value = p(wo, wi);
            return PhaseFunctionSample{
                .p = p_value,
                .wi = wi,
                .pdf = p_value,
                .valid = def(true)};
        }
        [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const override {
            return p(wo, wi);
        }

    public:
        explicit HenyeyGreensteinInstance(
            Pipeline &pipeline, const HenyeyGreenstein *phase_function) noexcept
            : PhaseFunction::Instance{pipeline, phase_function} {}
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return make_unique<HenyeyGreensteinInstance>(pipeline, this);
    }

public:
    HenyeyGreenstein(Scene *scene, const SceneNodeDesc *desc) noexcept
        : PhaseFunction{scene, desc},
          _g{clamp(desc->property_float_or_default("g", 0.f), -1.f, 1.f)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HenyeyGreenstein)