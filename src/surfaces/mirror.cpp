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

public:
    struct Params {
        TextureHandle color;
        TextureHandle roughness;
        bool remap_roughness;
        bool isotropic;
    };

private:
    const Texture *_color;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    MirrorSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default(
              "roughness", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", false)} {
        if (_color->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in MirrorSurface::color. [{}]",
                desc->source_location().string());
        }
        if (_roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in MirrorSurface::roughness. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        Params params{
            .color = *pipeline.encode_texture(command_buffer, _color),
            .roughness = *pipeline.encode_texture(command_buffer, _roughness),
            .remap_roughness = _remap_roughness,
            .isotropic = _roughness->channels() == 1u};
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<Params>(1u);
        command_buffer << buffer_view.copy_from(&params) << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::MirrorSurface::Params,
    color, roughness, remap_roughness, isotropic){};

namespace luisa::render {

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
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> refl, Expr<float2> alpha) noexcept
        : _it{it}, _swl{swl}, _fresnel{refl}, _distribution{alpha},
          _refl{refl, &_distribution, &_fresnel} {}
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _refl.evaluate(wo_local, wi_local);
        auto pdf = _refl.pdf(wo_local, wi_local);
        return {.swl = _swl, .f = f, .pdf = pdf,
                .alpha = _distribution.alpha(),
                .eta = make_float4(1.f)};
    }
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto u = sampler.generate_2d();
        auto f = _refl.sample(_it.wo_local(), &wi_local, u, &pdf);
        return {.wi = _it.shading().local_to_world(wi_local),
                .eval = {.swl = _swl, .f = f, .pdf = pdf,
                         .alpha = _distribution.alpha(),
                         .eta = make_float4(1.f)}};
    }

    void update() noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
    void backward(Pipeline &pipeline, const SampledWavelengths &swl_fixed, Expr<float4> k, Float learning_rate, Expr<float3> wi) noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
};

unique_ptr<Surface::Closure> MirrorSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<MirrorSurface::Params>(it.shape()->surface_buffer_id()).read(0u);
    auto R = pipeline.evaluate_color_texture(params.color, it, swl, time);
    auto r = pipeline.evaluate_generic_texture(params.roughness, it, time);
    auto roughness = ite(params.isotropic, r.xx(), r.xy());
    auto alpha = ite(
        params.remap_roughness,
        TrowbridgeReitzDistribution::roughness_to_alpha(roughness),
        roughness);
    return luisa::make_unique<MirrorClosure>(it, swl, R, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MirrorSurface)
