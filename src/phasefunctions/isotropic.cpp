//
// Created by ChenXin on 2023/2/15.
//

#include <base/phase_function.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class IsotropicPhaseFunction : public PhaseFunction {
public:
    class IsotropicInstance : public PhaseFunction::Instance {
    protected:
        friend class IsotropicPhaseFunction;

    public:
        [[nodiscard]] Float evaluate(Expr<float3> wo, Expr<float3> wi) const override {
            return 1.f / (4.f * pi);
        }
        [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u) const override {
            SampledDirection wi{
                .wi = sample_cosine_hemisphere(u),
                .pdf = 1.f / (4.f * pi),
            };
            return wi;
        }

    public:
        explicit IsotropicInstance(Pipeline &pipeline, const IsotropicPhaseFunction *phase_function) noexcept
            : PhaseFunction::Instance{pipeline, phase_function} {}
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return make_unique<IsotropicInstance>(pipeline, this);
    }

public:
    IsotropicPhaseFunction(Scene *scene, const SceneNodeDesc *desc) noexcept
        : PhaseFunction{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::IsotropicPhaseFunction)