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

class SubstrateSurface final : public Surface {

public:
    struct Params {
        TextureHandle Kd;
        TextureHandle Ks;
        TextureHandle roughness;
        bool remap_roughness;
        bool isotropic;
    };

private:
    const Texture *_kd;
    const Texture *_ks;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    SubstrateSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node_or_default(
              "Kd", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _ks{scene->load_texture(desc->property_node_or_default(
              "Ks", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default(
              "roughness", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (_kd->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in SubstrateSurface::Kd. [{}]",
                desc->source_location().string());
        }
        if (_ks->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in SubstrateSurface::Ks. [{}]",
                desc->source_location().string());
        }
        if (_roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in SubstrateSurface::roughness. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<Params>(1u);
        Params params{
            .Kd = *pipeline.encode_texture(command_buffer, _kd),
            .Ks = *pipeline.encode_texture(command_buffer, _ks),
            .roughness = *pipeline.encode_texture(command_buffer, _roughness),
            .remap_roughness = _remap_roughness,
            .isotropic = _roughness->channels() == 1u};
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
    luisa::render::SubstrateSurface::Params,
    Kd, Ks, roughness, remap_roughness, isotropic){};

namespace luisa::render {

class SubstrateClosure final : public Surface::Closure {

private:
    const Interaction &_interaction;
    TrowbridgeReitzDistribution _distribution;
    FresnelBlend _blend;

public:
    SubstrateClosure(const Interaction &it, Expr<float4> Kd, Expr<float4> Ks, Expr<float2> alpha) noexcept
        : _interaction{it}, _distribution{alpha}, _blend{Kd, Ks, &_distribution} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = _interaction.shading().world_to_local(wi);
        auto f = _blend.evaluate(wo_local, wi_local);
        auto pdf = _blend.evaluate(wo_local, wi_local);
        return {.f = f, .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto wi_local = def<float3>();
        auto f = _blend.sample(wo_local, &wi_local, u, &pdf);
        auto wi = _interaction.shading().local_to_world(wi_local);
        return {.wi = wi, .eval = {.f = f, .pdf = pdf}};
    }
};

luisa::unique_ptr<Surface::Closure> SubstrateSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<Params>(it.shape()->surface_buffer_id()).read(0u);
    auto Kd = pipeline.evaluate_color_texture(params.Kd, it, swl, time);
    auto Ks = pipeline.evaluate_color_texture(params.Ks, it, swl, time);
    auto r = pipeline.evaluate_generic_texture(params.roughness, it, time);
    auto roughness = ite(params.isotropic, r.xx(), r.xy());
    auto alpha = ite(
        params.remap_roughness,
        TrowbridgeReitzDistribution::roughness_to_alpha(roughness),
        roughness);
    return luisa::make_unique<SubstrateClosure>(it, Kd, Ks, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SubstrateSurface)
