//
// Created by ChenXin on 2023/3/8.
//

#include <base/medium.h>
#include <base/pipeline.h>

namespace luisa::render {

using compute::Ray;

class VacuumMedium : public Medium {

public:
    class VacuumMediumInstance;

    class VacuumMediumClosure : public Medium::Closure {

    private:
        [[nodiscard]] Sample _sample(Expr<float> t_max, Sampler::Instance *sampler) const noexcept override {
            instance()->pipeline().printer().error_with_location("VacuumMediumClosure::sample() is not implemented. Priority={}", instance()->priority());
            return Sample::zero(swl().dimension());
        }
        SampledSpectrum _transmittance(Expr<float> t, Sampler::Instance *sampler) const noexcept override {
            return {swl().dimension(), 1.0f};
        }

    public:
        VacuumMediumClosure(
            const VacuumMediumInstance *instance, Expr<Ray> ray, luisa::shared_ptr<Interaction> it,
            const SampledWavelengths &swl, Expr<float> time) noexcept
            : Medium::Closure{instance, ray, std::move(it), swl, time, 1.0f} {}

    };

    class VacuumMediumInstance : public Medium::Instance {

    private:
        friend class VacuumMedium;

    public:
        VacuumMediumInstance(const Pipeline &pipeline, const Medium *medium) noexcept
            : Medium::Instance(pipeline, medium) {}
        [[nodiscard]] luisa::unique_ptr<Closure> closure(
            Expr<Ray> ray, luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
            return luisa::make_unique<VacuumMediumClosure>(this, ray, std::move(it), swl, time);
        }
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<VacuumMediumInstance>(pipeline, this);
    }

public:
    VacuumMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc} {
        _priority = 0u;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VacuumMedium)