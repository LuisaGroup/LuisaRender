//
// Created by ChenXin on 2023/2/13.
//

#include <base/medium.h>
#include <base/phase_function.h>
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
            : _seg{t_min, t_max, sigma_maj}, _called(def(false)) {}

        [[nodiscard]] optional<RayMajorantSegment> next() noexcept override {
            luisa::optional<RayMajorantSegment> seg;
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

    private:
        SampledSpectrum _sigma_a;   // absorption coefficient
        SampledSpectrum _sigma_s;   // scattering coefficient
        SampledSpectrum _sigma_t;   // extinction coefficient
        SampledSpectrum _le;        // emission coefficient
        const PhaseFunction::Instance *_phase_function;

    public:
        [[nodiscard]] Sample sample(Expr<float> t_max, PCG32 &rng) const noexcept override {
            // TODO
            auto sample = Sample::zero(swl().dimension());

            // sample collision-free distance
            auto t = -log(max(1.f - rng.uniform_float(), 1e-6f)) / _sigma_t.average();
            sample.t = t;

            // hit volume boundary, no collision
            $if(t > t_max - RAY_EPSILON){
                sample.ray = ray();

                auto tr = transmittance(t_max, rng);
                auto p_surface = tr;
                auto pdf = p_surface;
                sample.eval.f = tr / pdf.sum();
                sample.is_scattered = false;
            }
            // in-scattering
            $else{
                // sample direction
                auto pf_sample = _phase_function->sample_p(
                    ray()->direction(), make_float2(rng.uniform_float(), rng.uniform_float()));
                sample.ray = make_ray(ray()->origin() + ray()->direction() * t, pf_sample.wi);

                auto tr = transmittance(t, rng);
                auto pdf_distance = _sigma_t * tr;
                auto pdf = pdf_distance;
                sample.eval.f = _sigma_s * tr / pdf.sum();
                sample.is_scattered = true;
            };

            return sample;
        }
        [[nodiscard]] SampledSpectrum transmittance(Expr<float> t, PCG32 &rng) const noexcept override {
            return analyticTransmittance(t, _sigma_t);
        }
        [[nodiscard]] unique_ptr<RayMajorantIterator> sample_iterator(Expr<float> t_max) const noexcept override {
            return luisa::make_unique<HomogeneousMajorantIterator>(0.f, t_max, _sigma_a + _sigma_s);
        }

    public:
        HomogeneousMediumClosure(
            const HomogeneousMediumInstance *instance, Expr<Ray> ray,
            const SampledWavelengths &swl, Expr<float> time, Expr<float> eta,
            const SampledSpectrum &sigma_a, const SampledSpectrum &sigma_s, const SampledSpectrum &le,
            const PhaseFunction::Instance *phase_function) noexcept
            : Medium::Closure{instance, ray, swl, time, eta},
              _sigma_a{sigma_a}, _sigma_s{sigma_s}, _sigma_t{sigma_a + sigma_s},
              _le{le}, _phase_function{phase_function} {}
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
            auto [sigma_a, strength_a] = _sigma_a != nullptr ? _sigma_a->evaluate_albedo_spectrum(it, swl, time) : Spectrum::Decode::one(swl.dimension());
            auto [sigma_s, strength_s] = _sigma_s != nullptr ? _sigma_s->evaluate_albedo_spectrum(it, swl, time) : Spectrum::Decode::one(swl.dimension());
            auto [Le, strength_Le] = _le != nullptr ? _le->evaluate_albedo_spectrum(it, swl, time) : Spectrum::Decode::one(swl.dimension());
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
        LUISA_ASSERT(_le != nullptr && _le->is_constant(), "Le must be specified as constant");
        LUISA_ASSERT(_phase_function != nullptr, "Phase function must be specified");
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HomogeneousMedium)