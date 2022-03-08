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
          _sigma{scene->load_texture(desc->property_node_or_default("sigma"))} {
        if (_kd->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in MatteSurface::Kd. [{}]",
                desc->source_location().string());
        }
        if (_sigma != nullptr && _sigma->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in MatteSurface::sigma. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MatteInstance final : public Surface::Instance {

private:
    const Texture::Instance *_kd;
    const Texture::Instance *_sigma;

public:
    MatteInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kd, const Texture::Instance *sigma) noexcept
        : Surface::Instance{pipeline, surface}, _kd{Kd}, _sigma{sigma} {}
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MatteSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto sigma = pipeline.build_texture(command_buffer, _sigma);
    return luisa::make_unique<MatteInstance>(pipeline, this, Kd, sigma);
}

class MatteClosure final : public Surface::Closure {

private:
    Float3 _wo_local;
    Frame _shading;
    const SampledWavelengths &_swl;
    OrenNayar _oren_nayar;

public:
    MatteClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> albedo, Expr<float> sigma) noexcept
        : Surface::Closure{instance},
          _wo_local{it.wo_local()}, _shading{it.shading()},
          _swl{swl}, _oren_nayar{albedo, sigma} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _wo_local;
        auto wi_local = _shading.world_to_local(wi);
        auto f = _oren_nayar.evaluate(wo_local, wi_local);
        auto pdf = _oren_nayar.pdf(wo_local, wi_local);
        return {.swl = _swl,
                .f = f,
                .pdf = pdf,
                .alpha = make_float2(1.f),
                .eta = make_float4(1.f)};
    }
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _wo_local;
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = _oren_nayar.sample(wo_local, &wi_local, u, &pdf);
        auto wi = _shading.local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.swl = _swl,
                         .f = f,
                         .pdf = pdf,
                         .alpha = make_float2(1.f),
                         .eta = make_float4(1.f)}};
    }
};

luisa::unique_ptr<Surface::Closure> MatteInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto Kd = _kd->evaluate(it, swl, time);
    auto sigma = _sigma ? clamp(_sigma->evaluate(it, swl, time).x, 0.f, 90.f) : 0.f;
    return luisa::make_unique<MatteClosure>(this, it, swl, Kd, sigma);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MatteSurface)
