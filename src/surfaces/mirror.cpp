//
// Created by Mike Smith on 2022/1/12.
//

#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class MirrorSurface final : public Surface {

private:
    const Texture *_color;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    MirrorSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", false)} {
        if (_color->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in MirrorSurface::color. [{}]",
                desc->source_location().string());
        }
        if (_roughness != nullptr && _roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in MirrorSurface::roughness. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MirrorInstance final : public Surface::Instance {

private:
    const Texture::Instance *_color;
    const Texture::Instance *_roughness;

public:
    MirrorInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *color, const Texture::Instance *roughness) noexcept
        : Surface::Instance{pipeline, surface}, _color{color}, _roughness{roughness} {}
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MirrorSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto color = pipeline.build_texture(command_buffer, _color);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<MirrorInstance>(pipeline, this, color, roughness);
}

using namespace luisa::compute;

class SchlickFresnel final : public Fresnel {

private:
    Float4 R0;

public:
    explicit SchlickFresnel(Expr<float4> R0) noexcept : R0{R0} {}
    [[nodiscard]] Float4 evaluate(Expr<float> cosI) const noexcept override {
        auto m = saturate(1.f - cosI);
        auto weight = sqr(sqr(m)) * m;
        return lerp(R0, 1.f, weight);
    }
};

class MirrorClosure final : public Surface::Closure {

private:
    const Interaction &_it;
    const SampledWavelengths &_swl;
    SchlickFresnel _fresnel;
    TrowbridgeReitzDistribution _distribution;
    MicrofacetReflection _refl;

public:
    MirrorClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> refl, Expr<float2> alpha) noexcept
        : Surface::Closure{instance}, _it{it}, _swl{swl},
          _fresnel{refl}, _distribution{alpha},
          _refl{refl, &_distribution, &_fresnel} {}
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _refl.evaluate(wo_local, wi_local);
        auto pdf = _refl.pdf(wo_local, wi_local);
        return {.swl = _swl,
                .f = f,
                .pdf = pdf,
                .alpha = _distribution.alpha(),
                .eta = make_float4(1.f)};
    }
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto u = sampler.generate_2d();
        auto f = _refl.sample(_it.wo_local(), &wi_local, u, &pdf);
        return {.wi = _it.shading().local_to_world(wi_local),
                .eval = {.swl = _swl,
                         .f = f,
                         .pdf = pdf,
                         .alpha = _distribution.alpha(),
                         .eta = make_float4(1.f)}};
    }
};

luisa::unique_ptr<Surface::Closure> MirrorInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<MirrorSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    auto color = _color->evaluate(it, swl, time);
    return luisa::make_unique<MirrorClosure>(this, it, swl, color, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MirrorSurface)
