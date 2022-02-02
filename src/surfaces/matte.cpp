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

class MatteSurface final : public Surface {

private:
    const Texture *_kd;
    const Texture *_sigma;

public:
    MatteSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node_or_default(
              "Kd", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _sigma{scene->load_texture(desc->property_node_or_default(
              "sigma", SceneNodeDesc::shared_default_texture("ConstGeneric")))} {
        if (_kd->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in MatteSurface::Kd. [{}]",
                desc->source_location().string());
        }
        if (_sigma->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in MatteSurface::sigma. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<TextureHandle>(2u);
        std::array textures{
            *pipeline.encode_texture(command_buffer, _kd),
            *pipeline.encode_texture(command_buffer, _sigma)};
        command_buffer << buffer_view.copy_from(textures.data())
                       << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class MatteClosure final : public Surface::Closure {

private:
    const Interaction &_interaction;
    const SampledWavelengths &_swl;
    OrenNayar _oren_nayar;

public:
    MatteClosure(const Interaction &it, const SampledWavelengths &swl,
                 Expr<float4> albedo, Expr<float> sigma) noexcept
        : _interaction{it}, _swl{swl}, _oren_nayar{albedo, sigma} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = _interaction.shading().world_to_local(wi);
        auto f = _oren_nayar.evaluate(wo_local, wi_local);
        auto pdf = _oren_nayar.pdf(wo_local, wi_local);
        return {.swl = _swl, .f = f, .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = _oren_nayar.sample(wo_local, &wi_local, u, &pdf);
        auto wi = _interaction.shading().local_to_world(wi_local);
        return {.wi = wi, .eval = {.swl = _swl, .f = f, .pdf = pdf}};
    }
};

luisa::unique_ptr<Surface::Closure> MatteSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto buffer = pipeline.buffer<TextureHandle>(it.shape()->surface_buffer_id());
    auto R = pipeline.evaluate_color_texture(buffer.read(0u), it, swl, time);
    auto sigma = pipeline.evaluate_generic_texture(buffer.read(1u), it, time).x;
    return luisa::make_unique<MatteClosure>(it, swl, R, clamp(sigma, 0.f, 90.f));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MatteSurface)
