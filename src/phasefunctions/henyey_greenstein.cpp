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
    private:
        Float _g;

    protected:
        friend class HenyeyGreenstein;

    public:
        [[nodiscard]] Float evaluate(Expr<float3> wo, Expr<float3> wi) const override {
            auto cosTheta = dot(wo, wi);
            auto denom = 1.f + _g * _g + 2.f * _g * cosTheta;
            return inv_pi / 4.f * (1.f - _g * _g) / (denom * sqrt(max(0.f, denom)));
        }
        [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u) const override {
            // sample cosTheta
            auto cosTheta = ite(
                abs(_g) < 1e-3f,
                1.f - 2.f * u.x,
                //        1.f + _g * _g - sqr((1.f - _g * _g) / (1 - _g + 2 * _g * u.x)) / (2.f * _g)
                -1.f / (2.f * _g) * (1.f + sqr(_g) - sqr((1.f - sqr(_g)) / (1.f + _g - 2.f * _g * u.x)))
            );

            // compute direction
            auto sinTheta = sqrt(max(0.f, 1.f - sqr(cosTheta)));
            auto phi = 2.f * pi * u.y;

            SampledDirection wi;
            wi.wi = make_float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
            wi.pdf = evaluate(wo, wi.wi);
            return wi;
        }

    public:
        explicit HenyeyGreensteinInstance(
            Pipeline &pipeline, const HenyeyGreenstein *phase_function, Expr<float> g) noexcept
            : PhaseFunction::Instance{pipeline, phase_function}, _g{g} {}
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return make_unique<HenyeyGreensteinInstance>(pipeline, this, _g);
    }

public:
    HenyeyGreenstein(Scene *scene, const SceneNodeDesc *desc) noexcept
        : PhaseFunction{scene, desc} {
        _g = desc->property_float_or_default("g", 0.f);
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HenyeyGreenstein)