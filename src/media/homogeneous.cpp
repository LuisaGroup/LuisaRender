//
// Created by ChenXin on 2023/2/13.
//

#include <base/medium.h>
#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class HomogeneousMedium : public Medium {

private:
    const float _eta;                    // eta of medium
    const Texture *_sigma_a;             // absorption coefficient
    const Texture *_sigma_s;             // scattering coefficient
    const Texture *_le;                  // emission coefficient
    const PhaseFunction *_phase_function;// phase function

public:
    class HomogeneousMajorantIterator : public RayMajorantIterator {
    public:
        explicit HomogeneousMajorantIterator(Float t_min, Float t_max, SampledSpectrum sigma_maj) noexcept
            : _seg{RayMajorantSegment{t_min, t_max, sigma_maj, false}}, _called(def(false)) {}

        [[nodiscard]] RayMajorantSegment next() noexcept override {
            RayMajorantSegment seg = RayMajorantSegment::one(0u);
            $if(!_called){
                seg = _seg;
                _called = true;
            };
            return seg;
        }

    private:
        RayMajorantSegment _seg;
        Bool _called;
    };

    class HomogeneousMediumInstance;

    class HomogeneousMediumClosure : public Medium::Closure {

    public:
        [[nodiscard]] SampledSpectrum transmittance(Expr<float> t, PCG32 &rng) const noexcept override {
            return analyticTransmittance(t, sigma_a() + sigma_s());
        }
        [[nodiscard]] unique_ptr<RayMajorantIterator> sample_iterator(Expr<float> t_max) const noexcept override {
            return luisa::make_unique<HomogeneousMajorantIterator>(0.f, t_max, sigma_a() + sigma_s());
        }

    public:
        HomogeneousMediumClosure(
            const HomogeneousMediumInstance *instance, Expr<Ray> ray,
            const SampledWavelengths &swl, Expr<float> time, Expr<float> eta,
            const SampledSpectrum &sigma_a, const SampledSpectrum &sigma_s, const SampledSpectrum &le,
            const PhaseFunction::Instance *phase_function) noexcept
            : Medium::Closure{instance, ray, swl, time, eta, sigma_a, sigma_s, le, phase_function} {}
    };

    class HomogeneousMediumInstance : public Medium::Instance {

    private:
        const Texture::Instance *_sigma_a;
        const Texture::Instance *_sigma_s;
        const Texture::Instance *_le;
        const PhaseFunction::Instance *_phase_function;

    protected:
        friend class HomogeneousMedium;

    public:
        HomogeneousMediumInstance(
            const Pipeline &pipeline, const Medium *medium,
            const Texture::Instance *sigma_a, const Texture::Instance *sigma_s,
            const Texture::Instance *Le, const PhaseFunction::Instance *phase_function) noexcept
            : Medium::Instance(pipeline, medium), _sigma_a{sigma_a}, _sigma_s{sigma_s}, _le{Le}, _phase_function{phase_function} {}
        [[nodiscard]] luisa::unique_ptr<Closure> closure(
            Expr<Ray> ray, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
            Interaction it;
            auto [sigma_a, strength_a] = _sigma_a->evaluate_albedo_spectrum(it, swl, time);
            auto [sigma_s, strength_s] = _sigma_s->evaluate_albedo_spectrum(it, swl, time);
            auto [Le, strength_Le] = _le != nullptr ? _le->evaluate_albedo_spectrum(it, swl, time) : Spectrum::Decode::zero(swl.dimension());
            return luisa::make_unique<HomogeneousMediumClosure>(
                this, ray, swl, time, node<HomogeneousMedium>()->_eta,
                sigma_a, sigma_s, Le, _phase_function);
        }
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto sigma_a = pipeline.build_texture(command_buffer, _sigma_a);
        auto sigma_s = pipeline.build_texture(command_buffer, _sigma_s);
        auto Le = pipeline.build_texture(command_buffer, _le);
        auto phase_function = pipeline.build_phasefunction(command_buffer, _phase_function);
        return luisa::make_unique<HomogeneousMediumInstance>(pipeline, this, sigma_a, sigma_s, Le, phase_function);
    }

public:
    HomogeneousMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc},
          _eta{desc->property_float_or_default("eta", 1.f)},
          _sigma_a{scene->load_texture(desc->property_node_or_default("sigma_a"))},
          _sigma_s{scene->load_texture(desc->property_node_or_default("sigma_s"))},
          _le{scene->load_texture(desc->property_node_or_default("Le"))},
          _phase_function{scene->load_phase_function(desc->property_node_or_default("phasefunction"))} {
        LUISA_ASSERT(_sigma_a != nullptr && _sigma_a->is_constant(), "sigma_a must be specified as constant");
        LUISA_ASSERT(_sigma_s != nullptr && _sigma_s->is_constant(), "sigma_s must be specified as constant");
        LUISA_ASSERT(_le == nullptr || _le->is_constant(), "Le must be null/constant");
        LUISA_ASSERT(_phase_function != nullptr, "Phase function must be specified");
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HomogeneousMedium)