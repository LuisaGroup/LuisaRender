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
            make_desc("BK7", make_float3(1.5128724797046567f, 1.518526426727177f, 1.5268028319190252f)),
            make_desc("BAF10", make_float3(1.663482689459752f, 1.6732858980501335f, 1.6882038140269822f)),
            make_desc("FK51A", make_float3(1.4837998115476296f, 1.487791013531385f, 1.4936507866294566f)),
            make_desc("LASF9", make_float3(1.8385554755145304f, 1.8562744195711076f, 1.884937209476662f)),
            make_desc("SF5", make_float3(1.6634157372231755f, 1.677424628301045f, 1.700294496924069f)),
            make_desc("SF10", make_float3(1.7170541479783665f, 1.734127433196915f, 1.7622659583540365f)),
            make_desc("SF11", make_float3(1.7713632012573233f, 1.7915573054146121f, 1.8258481620747222f)),
            make_desc("Diamond", make_float3(2.4076888208234286f, 2.421389479656345f, 2.4442593103910073f)),
            make_desc("Ice", make_float3(1.3067738983564134f, 1.3110361167927982f, 1.3167829558780912f)),
            make_desc("Quartz", make_float3(1.4552432604483285f, 1.4599273927275231f, 1.4667353873623656f)),
            make_desc("SiO2", make_float3(1.4552432604483285f, 1.4599273927275231f, 1.4667353873623656f)),
            make_desc("Salt", make_float3(1.538654377305543f, 1.5471867947656404f, 1.560368730283437f)),
            make_desc("NaCl", make_float3(1.538654377305543f, 1.5471867947656404f, 1.560368730283437f)),
            make_desc("Sapphire", make_float3(1.7630793084415752f, 1.7706125009330247f, 1.7812914892330447f)),
            make_desc("Al2O3", make_float3(1.7630793084415752f, 1.7706125009330247f, 1.7812914892330447f))};
    }();
    for (auto &c : name) { c = static_cast<char>(tolower(c)); }
    auto iter = nodes.find(name);
    return iter == nodes.cend() ? nullptr : iter->second.get();
}

class GlassSurface final : public Surface {

private:
    const Texture *_kr;
    const Texture *_kt;
    const Texture *_roughness{nullptr};
    const Texture *_eta{nullptr};
    bool _remap_roughness;

public:
    GlassSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kr{scene->load_texture(desc->property_node_or_default(
              "Kr", SceneNodeDesc::shared_default_texture("Constant")))},
          _kt{scene->load_texture(desc->property_node_or_default(
              "Kt", SceneNodeDesc::shared_default_texture("Constant")))},
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
                if (_eta->channels() == 2u) [[unlikely]] {
                    LUISA_ERROR(
                        "Invalid channel count {} "
                        "for GlassSurface::eta.",
                        desc->source_location().string());
                }
            }
        }
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GlassInstance final : public Surface::Instance {

private:
    const Texture::Instance *_kr;
    const Texture::Instance *_kt;
    const Texture::Instance *_roughness{nullptr};
    const Texture::Instance *_eta{nullptr};

public:
    GlassInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kr, const Texture::Instance *Kt,
        const Texture::Instance *roughness, const Texture::Instance *eta) noexcept
        : Surface::Instance{pipeline, surface}, _kr{Kr}, _kt{Kt}, _roughness{roughness}, _eta{eta} {}
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> GlassSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kr = pipeline.build_texture(command_buffer, _kr);
    auto Kt = pipeline.build_texture(command_buffer, _kt);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto eta = pipeline.build_texture(command_buffer, _eta);
    return luisa::make_unique<GlassInstance>(pipeline, this, Kr, Kt, roughness, eta);
}

class GlassClosure final : public Surface::Closure {

private:
    SampledSpectrum _eta_i;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distribution;
    luisa::unique_ptr<FresnelDielectric> _fresnel;
    luisa::unique_ptr<MicrofacetReflection> _refl;
    luisa::unique_ptr<MicrofacetTransmission> _trans;
    Float _kr_ratio;

public:
    GlassClosure(
        const Surface::Instance *instance, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &Kr, const SampledSpectrum &Kt,
        const SampledSpectrum &eta, Expr<float2> alpha, Expr<float> Kr_ratio) noexcept
        : Surface::Closure{instance, it, swl, time}, _eta_i{swl.dimension(), 1.f},
          _distribution{luisa::make_unique<TrowbridgeReitzDistribution>(alpha)},
          _fresnel{luisa::make_unique<FresnelDielectric>(_eta_i, eta)},
          _refl{luisa::make_unique<MicrofacetReflection>(Kr, _distribution.get(), _fresnel.get())},
          _trans{luisa::make_unique<MicrofacetTransmission>(Kt, _distribution.get(), _eta_i, eta)},
          _kr_ratio{Kr_ratio} {}
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        SampledSpectrum f{_swl.dimension()};
        auto pdf = def(0.f);
        auto swl = _swl;
        $if(same_hemisphere(wo_local, wi_local)) {
            f = _refl->evaluate(wo_local, wi_local);
            pdf = _refl->pdf(wo_local, wi_local) * _kr_ratio;
        }
        $else {
            f = _trans->evaluate(wo_local, wi_local);
            pdf = _trans->pdf(wo_local, wi_local) * (1.f - _kr_ratio);
        };
        SampledSpectrum eta{_swl.dimension()};
        auto entering = wi_local.z < 0.f;
        for (auto i = 0u; i < eta.dimension(); i++) {
            eta[i] = ite(entering, _fresnel->eta_t()[i], 1.f);
        }
        return {.f = f, .pdf = pdf, .alpha = _distribution->alpha(), .eta = eta};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _it.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = SampledSpectrum{_swl.dimension()};
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto lobe = cast<int>(u.x >= _kr_ratio);
        auto swl = _swl;
        $if(lobe == 0u) {// Reflection
            u.x = u.x / _kr_ratio;
            f = _refl->sample(wo_local, &wi_local, u, &pdf);
            pdf *= _kr_ratio;
        }
        $else {// Transmission
            u.x = (u.x - _kr_ratio) / (1.f - _kr_ratio);
            f = _trans->sample(wo_local, &wi_local, u, &pdf);
            pdf *= (1.f - _kr_ratio);
        };
        auto wi = _it.shading().local_to_world(wi_local);
        SampledSpectrum eta{_swl.dimension()};
        auto entering = wi_local.z < 0.f;
        for (auto i = 0u; i < eta.dimension(); i++) {
            eta[i] = ite(entering, _fresnel->eta_t()[i], 1.f);
        }
        return {.wi = wi, .eval = {.f = f, .pdf = pdf, .alpha = _distribution->alpha(), .eta = eta}};
    }

    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

luisa::unique_ptr<Surface::Closure> GlassInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto Kr_rgb = saturate(_kr->evaluate(it, time).xyz());
    auto Kt_rgb = saturate(_kt->evaluate(it, time).xyz());
    auto Kr_lum = srgb_to_cie_y(Kr_rgb);
    auto Kt_lum = srgb_to_cie_y(Kt_rgb);
    auto Kr = swl.albedo_from_srgb(Kr_rgb);
    auto Kt = swl.albedo_from_srgb(Kt_rgb);
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, time);
        auto remap = node<GlassSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    SampledSpectrum eta{swl.dimension(), 1.5f};
    if (_eta != nullptr) {
        if (_eta->node()->channels() == 1u) {
            auto e = _eta->evaluate(it, time).x;
            for (auto i = 0u; i < eta.dimension(); i++) { eta[i] = e; }
        } else {
            auto e = _eta->evaluate(it, time).xyz();
            auto inv_bb = sqr(1.f / make_float3(700.0f, 546.1f, 435.8f));
            auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
            auto c = inverse(m) * e;
            for (auto i = 0u; i < swl.dimension(); i++) {
                auto inv_ll = sqr(1.f / swl.lambda(i));
                eta[i] = c.x + c.y * inv_ll + c.z * sqr(inv_ll);
            }
        }
    }
    auto Fr = fresnel_dielectric(cos_theta(it.wo_local()), 1.f, eta.average());
    auto Kr_ratio = ite(Kr_lum == 0.f, 0.f, Kr_lum / (Kr_lum + Kt_lum));
    return luisa::make_unique<GlassClosure>(
        this, it, swl, time, Kr, Kt, eta,
        alpha, clamp(Fr * Kr_ratio, 0.1f, 0.9f));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GlassSurface)
