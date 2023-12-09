//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

inline auto builtin_ior_texture_desc(luisa::string name) noexcept {
    static const auto nodes = [] {
        auto make_desc = [](luisa::string name, float3 ior) noexcept {
            auto desc = luisa::make_shared<SceneNodeDesc>(
                luisa::format("__glass_surface_builtin_ior_{}", name),
                SceneNodeTag::TEXTURE);
            desc->define(SceneNodeTag::TEXTURE, "Constant", {});
            desc->add_property("v", SceneNodeDesc::number_list{ior.x, ior.y, ior.z});
            for (auto &c : name) { c = static_cast<char>(tolower(c)); }
            return eastl::make_pair(std::move(name), std::move(desc));
        };
        using namespace std::string_view_literals;
        return luisa::fixed_map<luisa::string, luisa::shared_ptr<SceneNodeDesc>, 15>{
            make_desc("bk7", make_float3(1.5140814565098806f, 1.5165571794092296f, 1.5223224896834853f)),
            make_desc("baf10", make_float3(1.665552211440938f, 1.6698355055693541f, 1.680044942398477f)),
            make_desc("fk51a", make_float3(1.4846524304153899f, 1.486399794903804f, 1.4904761200647965f)),
            make_desc("lasf9", make_float3(1.8422161861952726f, 1.8499302872507852f, 1.8690214187351977f)),
            make_desc("sf5", make_float3(1.6663001504164476f, 1.6723956342450994f, 1.6875677127863755f)),
            make_desc("sf10", make_float3(1.720557014155419f, 1.7279815121138318f, 1.7465722778961204f)),
            make_desc("sf11", make_float3(1.7754589288508518f, 1.7842240428294434f, 1.8065917880168352f)),
            make_desc("diamond", make_float3(2.410486117067883f, 2.4164392529553234f, 2.431466411471524f)),
            make_desc("ice", make_float3(1.3077084260466776f, 1.3095827995357034f, 1.3137441348487024f)),
            make_desc("quartz", make_float3(1.4562471554155727f, 1.4582990183632742f, 1.4630571022260817f)),
            make_desc("salt", make_float3(1.5404463273409252f, 1.5441711411436845f, 1.5531314007749342f)),
            make_desc("sapphire", make_float3(1.764706495252994f, 1.7680107911479397f, 1.7755615929437936f))};
    }();
    for (auto &c : name) { c = static_cast<char>(tolower(c)); }
    auto iter = nodes.find(name);
    return iter == nodes.cend() ? nullptr : iter->second.get();
}

class GlassSurface : public Surface {

private:
    const Texture *_kr;
    const Texture *_kt;
    const Texture *_roughness;
    const Texture *_eta;
    bool _remap_roughness;

public:
    GlassSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kr{scene->load_texture(desc->property_node_or_default("Kr"))},
          _kt{scene->load_texture(desc->property_node_or_default("Kt"))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (auto eta_name = desc->property_string_or_default("eta"); !eta_name.empty()) {
            _eta = scene->load_texture(builtin_ior_texture_desc(eta_name));
            if (_eta == nullptr) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown built-in glass '{}'. "
                    "Fallback to constant IOR = 1.5. [{}]",
                    eta_name, desc->source_location().string());
            }
        } else {
            if ((_eta = scene->load_texture(desc->property_node_or_default("eta")))) {
                if (_eta->channels() == 2u || _eta->channels() == 4u) [[unlikely]] {
                    LUISA_ERROR(
                        "Invalid channel count {} "
                        "for GlassSurface::eta.",
                        desc->source_location().string());
                }
            }
        }
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override {
        return property_reflective | property_transmissive | property_differentiable;
    }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GlassInstance : public Surface::Instance {

public:
private:
    const Texture::Instance *_kr;
    const Texture::Instance *_kt;
    const Texture::Instance *_roughness;
    const Texture::Instance *_eta;

public:
    GlassInstance(const Pipeline &pipeline, const Surface *surface,
                  const Texture::Instance *Kr, const Texture::Instance *Kt,
                  const Texture::Instance *roughness, const Texture::Instance *eta) noexcept
        : Surface::Instance{pipeline, surface}, _kr{Kr}, _kt{Kt}, _roughness{roughness}, _eta{eta} {}
    [[nodiscard]] auto Kr() const noexcept { return _kr; }
    [[nodiscard]] auto Kt() const noexcept { return _kt; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }
    [[nodiscard]] auto eta() const noexcept { return _eta; }

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure *closure, const Interaction &it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
    [[nodiscard]] luisa::string closure_identifier() const noexcept override {
        return luisa::format("glass<{}, {}, {}, {}>",
                             Texture::Instance::diff_param_identifier(_kr),
                             Texture::Instance::diff_param_identifier(_kt),
                             Texture::Instance::diff_param_identifier(_roughness),
                             Texture::Instance::diff_param_identifier(_eta));
    }
};

luisa::unique_ptr<Surface::Instance> GlassSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kr = pipeline.build_texture(command_buffer, _kr);
    auto Kt = pipeline.build_texture(command_buffer, _kt);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto eta = pipeline.build_texture(command_buffer, _eta);
    return luisa::make_unique<GlassInstance>(pipeline, this, Kr, Kt, roughness, eta);
}

class GlassClosure : public Surface::Closure {

public:
    struct Context {
        Interaction it;
        SampledSpectrum Kr;
        SampledSpectrum Kt;
        Float eta_i;
        Float eta_t;
        Bool dispersive;
        Float2 alpha;
        Float Kr_ratio;
    };

public:
    using Surface::Closure::Closure;

    [[nodiscard]] SampledSpectrum albedo() const noexcept override {
        return context<Context>().Kr;
    }
    [[nodiscard]] Float2 roughness() const noexcept override {
        return TrowbridgeReitzDistribution::alpha_to_roughness(
            context<Context>().alpha);
    }
    [[nodiscard]] optional<Float> eta() const noexcept override {
        return context<Context>().eta_t;
    }
    [[nodiscard]] luisa::optional<Bool> is_dispersive() const noexcept override {
        return context<Context>().dispersive;
    }
    [[nodiscard]] const Interaction &it() const noexcept override { return context<Context>().it; }

private:
    [[nodiscard]] auto _refl_prob(const FresnelDielectric &fresnel, Expr<float> kr_ratio, Expr<float3> wo) const noexcept {
        auto F = fresnel.evaluate(cos_theta(wo))[0u];
        auto r = kr_ratio * F;
        auto t = (1.f - kr_ratio) * (1.f - F);
        return ite(r == 0.f, 0.f, r / (r + t));
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo,
                                                Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        auto &ctx = context<Context>();
        auto &it = ctx.it;
        auto distribution = TrowbridgeReitzDistribution{ctx.alpha};
        auto fresnel = FresnelDielectric{ctx.eta_i, ctx.eta_t};
        auto refl = MicrofacetReflection{ctx.Kr, &distribution, &fresnel};
        auto trans = MicrofacetTransmission{ctx.Kt, &distribution, ctx.eta_i, ctx.eta_t};

        auto wo_local = it.shading().world_to_local(wo);
        auto wi_local = it.shading().world_to_local(wi);
        SampledSpectrum f{swl().dimension()};
        auto pdf = def(0.f);
        auto ratio = _refl_prob(fresnel, ctx.Kr_ratio, wo_local);
        $if(same_hemisphere(wo_local, wi_local)) {
            f = refl.evaluate(wo_local, wi_local, mode);
            pdf = refl.pdf(wo_local, wi_local, mode) * ratio;
        }
        $else {
            f = trans.evaluate(wo_local, wi_local, mode);
            pdf = trans.pdf(wo_local, wi_local, mode) * (1.f - ratio);
        };
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo,
                                          Expr<float> u_lobe, Expr<float2> u,
                                          TransportMode mode) const noexcept override {
        auto &ctx = context<Context>();
        auto &it = ctx.it;
        auto distribution = TrowbridgeReitzDistribution{ctx.alpha};
        auto fresnel = FresnelDielectric{ctx.eta_i, ctx.eta_t};
        auto refl = MicrofacetReflection{ctx.Kr, &distribution, &fresnel};
        auto trans = MicrofacetTransmission{ctx.Kt, &distribution, ctx.eta_i, ctx.eta_t};

        auto wo_local = it.shading().world_to_local(wo);
        auto pdf = def(0.f);
        auto f = SampledSpectrum{swl().dimension()};
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto event = def(Surface::event_reflect);
        auto ratio = _refl_prob(fresnel, ctx.Kr_ratio, wo_local);
        auto wi = def(make_float3());
        $if(u_lobe < ratio) {// Reflection
            f = refl.sample(wo_local, std::addressof(wi_local),
                            u, std::addressof(pdf), mode);
            wi = it.shading().local_to_world(wi_local);
            pdf *= ratio;
        }
        $else {// Transmission
            f = trans.sample(wo_local, std::addressof(wi_local),
                             u, std::addressof(pdf), mode);
            wi = it.shading().local_to_world(wi_local);
            pdf *= (1.f - ratio);
            event = ite(cos_theta(wo_local) > 0.f, Surface::event_enter, Surface::event_exit);
        };
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf}, .wi = wi, .event = event};
    }

    void _backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df_in,
                   TransportMode mode) const noexcept override {
        auto _instance = instance<GlassInstance>();

        auto &ctx = context<Context>();
        auto &it = ctx.it;
        auto distribution = TrowbridgeReitzDistribution{ctx.alpha};
        auto fresnel = FresnelDielectric{ctx.eta_i, ctx.eta_t};
        auto refl = MicrofacetReflection{ctx.Kr, &distribution, &fresnel};
        auto trans = MicrofacetTransmission{ctx.Kt, &distribution, ctx.eta_i, ctx.eta_t};

        auto wo_local = it.shading().world_to_local(wo);
        auto wi_local = it.shading().world_to_local(wi);
        auto df = df_in * abs_cos_theta(wi_local);

        Float2 d_alpha;

        $if(same_hemisphere(wo_local, wi_local)) {
            // Kr
            if (_instance->Kr() && _instance->Kr()->node()->requires_gradients()) {
                auto d_f = refl.backward(wo_local, wi_local, df, mode);
                d_alpha = d_f.dAlpha;
                _instance->Kr()->backward_albedo_spectrum(it, swl(), time(), zero_if_any_nan(d_f.dR));
            }
        }
        $else {
            // Ks
            if (_instance->Kt() && _instance->Kt()->node()->requires_gradients()) {
                auto d_f = trans.backward(wo_local, wi_local, df, mode);
                d_alpha = d_f.dAlpha;
                _instance->Kt()->backward_albedo_spectrum(it, swl(), time(), zero_if_any_nan(d_f.dT));
            }
        };

        // roughness
        if (auto roughness = _instance->roughness();
            roughness != nullptr && roughness->node()->requires_gradients()) {
            auto remap = _instance->node<GlassSurface>()->remap_roughness();
            auto r_f4 = roughness->evaluate(it, swl(), time());
            auto r = roughness->node()->channels() == 1u ? r_f4.xx() : r_f4.xy();
            auto grad_alpha_roughness = [](auto &&x) noexcept {
                return TrowbridgeReitzDistribution::grad_alpha_roughness(x);
            };
            auto d_r = d_alpha * (remap ? grad_alpha_roughness(r) : make_float2(1.f));
            auto d_r_f4 = roughness->node()->channels() == 1u ?
                              make_float4(d_r.x + d_r.y, 0.f, 0.f, 0.f) :
                              make_float4(d_r, 0.f, 0.f);
            auto roughness_grad_range = 5.f * (roughness->node()->range().y - roughness->node()->range().x);
            roughness->backward(it, swl(), time(),
                                ite(any(isnan(d_r_f4) || abs(d_r_f4) > roughness_grad_range), 0.f, d_r_f4));
        }
    }
};

luisa::unique_ptr<Surface::Closure> GlassInstance::create_closure(
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<GlassClosure>(this, pipeline(), swl, time);
}

void GlassInstance::populate_closure(Surface::Closure *closure, const Interaction &it,
                                     Expr<float3> wo, Expr<float> eta_i) const noexcept {

    auto alpha = def(make_float2(0.f));
    auto &swl = closure->swl();
    auto time = closure->time();
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<GlassSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }

    // Kr, Kt
    auto [Kr, Kr_lum] = _kr ? _kr->evaluate_albedo_spectrum(it, swl, time) :
                              Spectrum::Decode::one(swl.dimension());
    auto [Kt, Kt_lum] = _kt ? _kt->evaluate_albedo_spectrum(it, swl, time) :
                              Spectrum::Decode::one(swl.dimension());
    auto Kr_ratio = ite(Kr_lum == 0.f, 0.f, Kr_lum / (Kr_lum + Kt_lum));

    // eta
    auto eta = def(1.5f);
    auto dispersive = def(false);
    if (_eta != nullptr) {
        if (_eta->node()->channels() == 1u ||
            pipeline().spectrum()->node()->is_fixed()) {
            eta = _eta->evaluate(it, swl, time).x;
        } else {
            auto e = _eta->evaluate(it, swl, time).xyz();
            auto inv_bb = sqr(1.f / fraunhofer_wavelengths);
            auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
            auto c = inverse(m) * e;
            auto inv_ll = sqr(1.f / swl.lambda(0u));
            eta = c.x + c.y * inv_ll + c.z * sqr(inv_ll);
            dispersive = !(e.x == e.y & e.y == e.z);
        }
    }

    GlassClosure::Context ctx{
        .it = it,
        .Kr = Kr,
        .Kt = Kt,
        .eta_i = eta_i,
        .eta_t = eta,
        .dispersive = dispersive,
        .alpha = alpha,
        .Kr_ratio = Kr_ratio};

    closure->bind(std::move(ctx));
}

using NormalMapGlassSurface = NormalMapWrapper<
    GlassSurface, GlassInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapGlassSurface)
