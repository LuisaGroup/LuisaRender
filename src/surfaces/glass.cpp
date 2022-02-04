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

public:
    struct Params {
        TextureHandle Kr;
        TextureHandle Kt;
        TextureHandle roughness;
        TextureHandle eta;
        bool remap_roughness;
        bool isotropic;
        bool dispersion;
    };

private:
    const Texture *_kr;
    const Texture *_kt;
    const Texture *_roughness;
    const Texture *_eta;
    bool _remap_roughness;
    luisa::unique_ptr<SceneNodeDesc> _builtin_ior;

public:
    GlassSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kr{scene->load_texture(desc->property_node_or_default(
              "Kr", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _kt{scene->load_texture(desc->property_node_or_default(
              "Kt", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default(
              "roughness", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
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
        if (_roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in GlassSurface::roughness. [{}]",
                desc->source_location().string());
        }
        if (auto eta_name = desc->property_string_or_default("eta"); !eta_name.empty()) {
            _eta = scene->load_texture(builtin_ior_texture_desc(eta_name));
            if (_eta == nullptr) [[unlikely]] {
                LUISA_ERROR(
                    "Unknown built-in glass '{}'. [{}]",
                    eta_name, desc->source_location().string());
            }
        } else {
            _eta = scene->load_texture(desc->property_node_or_default(
                "eta", SceneNodeDesc::shared_default_texture("ConstGeneric")));
        }
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
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<Params>(1u);
        Params params{
            .Kr = *pipeline.encode_texture(command_buffer, _kr),
            .Kt = *pipeline.encode_texture(command_buffer, _kt),
            .roughness = *pipeline.encode_texture(command_buffer, _roughness),
            .eta = *pipeline.encode_texture(command_buffer, _eta),
            .remap_roughness = _remap_roughness,
            .isotropic = _roughness->channels() == 1u,
            .dispersion = _eta->channels() != 1u};
        command_buffer << buffer_view.copy_from(&params)
                       << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::GlassSurface::Params,
    Kr, Kt, roughness, eta,
    remap_roughness, isotropic, dispersion){};

namespace luisa::render {

class GlassClosure final : public Surface::Closure {

private:
    const Interaction &_interaction;
    const SampledWavelengths &_swl;
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
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> eta, Expr<float4> Kr, Expr<float4> Kt,
        Expr<float2> alpha, Expr<float> Kr_ratio) noexcept
        : _interaction{it}, _swl{swl}, _distribution{alpha}, _fresnel{make_float4(1.0f), eta},
          _refl{Kr, &_distribution, &_fresnel}, _trans{Kt, &_distribution, make_float4(1.f), eta},
          _kr_ratio{Kr_ratio}, _dispersion{has_dispersion(eta)} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = _interaction.shading().world_to_local(wi);
        auto f = def<float4>();
        auto pdf = def(0.f);
        auto swl = _swl;
        auto t = saturate(abs(_fresnel.evaluate(cos_theta(wo_local)).x) * _kr_ratio);
        $if(same_hemisphere(wo_local, wi_local)) {
            f = _refl.evaluate(wo_local, wi_local);
            pdf = _refl.pdf(wo_local, wi_local) * t;
        }
        $else {
            f = _trans.evaluate(wo_local, wi_local);
            pdf = _trans.pdf(wo_local, wi_local) * (1.f - t);
            $if(_dispersion) { swl.terminate_secondary(); };
        };
        return {.swl = swl, .f = f, .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = def<float4>();
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
        auto wi = _interaction.shading().local_to_world(wi_local);
        return {.wi = wi, .eval = {.swl = swl, .f = f, .pdf = pdf}};
    }
};

luisa::unique_ptr<Surface::Closure> GlassSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<Params>(it.shape()->surface_buffer_id()).read(0u);
    auto Kr = pipeline.evaluate_color_texture(params.Kr, it, swl, time);
    auto Kt = sqrt(pipeline.evaluate_color_texture(params.Kt, it, swl, time));
    auto r = pipeline.evaluate_generic_texture(params.roughness, it, time);
    auto e = pipeline.evaluate_generic_texture(params.eta, it, time);
    auto eta_basis = ite(params.dispersion, e.xyz(), ite(e.x == 0.f, 1.5f, e.x));
    auto roughness = ite(params.isotropic, r.xx(), r.xy());
    auto alpha = ite(
        params.remap_roughness,
        TrowbridgeReitzDistribution::roughness_to_alpha(roughness),
        roughness);
    auto Kr_lum = swl.cie_y(Kr);
    auto Kt_lum = swl.cie_y(Kt);
    auto Kr_ratio = ite(Kr_lum == 0.f, 0.f, Kr_lum / (Kr_lum + Kt_lum));
    // interpolate eta using Cauchy's dispersion formula
    auto inv_bb = sqr(1.f / make_float3(700.0f, 546.1f, 435.8f));
    auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
    auto c = inverse(m) * eta_basis;
    auto inv_ll = sqr(1.f / swl.lambda());
    auto eta = make_float4(
        dot(c, make_float3(1.f, inv_ll.x, sqr(inv_ll.x))),
        dot(c, make_float3(1.f, inv_ll.y, sqr(inv_ll.y))),
        dot(c, make_float3(1.f, inv_ll.z, sqr(inv_ll.z))),
        dot(c, make_float3(1.f, inv_ll.w, sqr(inv_ll.w))));
    return luisa::make_unique<GlassClosure>(
        it, swl, eta, Kr, Kt, alpha, Kr_ratio);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GlassSurface)
