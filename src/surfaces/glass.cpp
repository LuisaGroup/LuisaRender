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
            desc->define(SceneNodeTag::TEXTURE, "ConstGeneric", {});
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
              "Kr", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _kt{scene->load_texture(desc->property_node_or_default(
              "Kt", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (_kr->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in GlassSurface::Kr. [{}]",
                desc->source_location().string());
        }
        if (_kt->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in GlassSurface::Kt. [{}]",
                desc->source_location().string());
        }
        if (_roughness != nullptr &&
            _roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in GlassSurface::roughness. [{}]",
                desc->source_location().string());
        }
        if (auto eta_name = desc->property_string_or_default("eta"); !eta_name.empty()) {
            _eta = scene->load_texture(builtin_ior_texture_desc(eta_name));
            if (_eta == nullptr) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown built-in glass '{}'. "
                    "Fallback to constant IOR = 1.5. [{}]",
                    eta_name, desc->source_location().string());
            }
        } else {
            _eta = scene->load_texture(desc->property_node_or_default("eta"));
        }
        if (_eta != nullptr) {
            if (_eta->category() != Texture::Category::GENERIC) [[unlikely]] {
                LUISA_ERROR(
                    "Non-generic textures are not "
                    "allowed in GlassSurface::eta. [{}]",
                    desc->source_location().string());
            }
            if (_eta->channels() == 2u) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid channel count {} "
                    "for GlassSurface::eta.",
                    desc->source_location().string());
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
    TrowbridgeReitzDistribution _distribution;
    FresnelDielectric _fresnel;
    MicrofacetReflection _refl;
    MicrofacetTransmission _trans;
    Float _kr_ratio;
    Bool _dispersion;

    [[nodiscard]] static auto has_dispersion(auto eta) noexcept {
        return !(abs(eta.x - eta.y) < 1e-6f &
                 abs(eta.y - eta.z) < 1e-6f &
                 abs(eta.z - eta.w) < 1e-6f);
    }

public:
    GlassClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        Expr<float4> eta, Expr<float4> Kr, Expr<float4> Kt,
        Expr<float2> alpha, Expr<float> Kr_ratio) noexcept
        : Surface::Closure{instance, it, swl, time},
          _distribution{alpha}, _fresnel{make_float4(1.0f), eta},
          _refl{Kr, &_distribution, &_fresnel},
          _trans{Kt, &_distribution, make_float4(1.f), eta},
          _kr_ratio{Kr_ratio}, _dispersion{has_dispersion(eta)} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = def(make_float4());
        auto pdf = def(0.f);
        auto swl = _swl;
        auto t = clamp(_fresnel.evaluate(cos_theta(wo_local)).x, 0.2f, 0.8f) * _kr_ratio;
        $if(same_hemisphere(wo_local, wi_local)) {
            f = _refl.evaluate(wo_local, wi_local);
            pdf = _refl.pdf(wo_local, wi_local) * t;
        }
        $else {
            f = _trans.evaluate(wo_local, wi_local);
            pdf = _trans.pdf(wo_local, wi_local) * (1.f - t);
            $if(_dispersion) { swl.terminate_secondary(); };
        };
        return {.swl = swl, .f = f, .pdf = pdf, .alpha = _distribution.alpha(), .eta = ite(wi_local.z > 0.f, _fresnel.eta_i(), _fresnel.eta_t())};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _it.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = def(make_float4());
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto t = saturate(_fresnel.evaluate(cos_theta(wo_local)).x * _kr_ratio);
        auto lobe = cast<int>(u.x >= t);
        auto swl = _swl;
        $if(lobe == 0u) {// Reflection
            u.x = u.x / t;
            f = _refl.sample(wo_local, &wi_local, u, &pdf);
            pdf *= t;
        }
        $else {// Transmission
            u.x = (u.x - t) / (1.f - t);
            f = _trans.sample(wo_local, &wi_local, u, &pdf);
            pdf *= (1.f - t);
            $if(_dispersion) { swl.terminate_secondary(); };
        };
        auto wi = _it.shading().local_to_world(wi_local);
        auto eta = ite(wi_local.z > 0.f, _fresnel.eta_i(), _fresnel.eta_t());
        return {.wi = wi, .eval = {.swl = swl, .f = f, .pdf = pdf, .alpha = _distribution.alpha(), .eta = eta}};
    }

    void backward(Expr<float3> wi, Expr<float4> grad) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
};

luisa::unique_ptr<Surface::Closure> GlassInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto Kr = _kr->evaluate(it, swl, time);
    auto Kt = _kt->evaluate(it, swl, time);
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<GlassSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    auto eta = def(make_float4(1.5f));
    if (_eta != nullptr) {
        if (_eta->node()->channels() == 1u) {
            eta = _eta->evaluate(it, swl, time).xxxx();
        } else {
            auto e = _eta->evaluate(it, swl, time).xyz();
            auto inv_bb = sqr(1.f / make_float3(700.0f, 546.1f, 435.8f));
            auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
            auto c = inverse(m) * e;
            auto inv_ll = sqr(1.f / swl.lambda());
            eta = make_float4(
                dot(c, make_float3(1.f, inv_ll.x, sqr(inv_ll.x))),
                dot(c, make_float3(1.f, inv_ll.y, sqr(inv_ll.y))),
                dot(c, make_float3(1.f, inv_ll.z, sqr(inv_ll.z))),
                dot(c, make_float3(1.f, inv_ll.w, sqr(inv_ll.w))));
        }
    }
    auto Kr_lum = swl.cie_y(Kr);
    auto Kt_lum = swl.cie_y(Kt);
    auto Kr_ratio = ite(Kr_lum == 0.f, 0.f, Kr_lum / (Kr_lum + Kt_lum));
    return luisa::make_unique<GlassClosure>(
        this, it, swl, time, eta, Kr, Kt, alpha, Kr_ratio);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GlassSurface)
