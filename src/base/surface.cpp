//
// Created by Mike on 2021/12/14.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/interaction.h>
#include <base/pipeline.h>

namespace luisa::render {

Surface::Surface(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SURFACE},
      _alpha{scene->load_texture(desc->property_node_or_default("alpha"))} {
    LUISA_RENDER_CHECK_GENERIC_TEXTURE(Surface, alpha, 1);
}

luisa::unique_ptr<Surface::Instance> Surface::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto instance = _build(pipeline, command_buffer);
    instance->_alpha = pipeline.build_texture(command_buffer, _alpha);
    return instance;
}

Surface::Closure::Closure(
    const Surface::Instance *instance, Interaction it,
    const SampledWavelengths &swl, Expr<float> time) noexcept
    : _instance{instance}, _it{std::move(it)}, _swl{swl}, _time{time} {}

luisa::optional<Float> Surface::Closure::opacity() const noexcept {
    luisa::optional<Float> o;
    if (auto alpha = _instance->alpha()) {
        o.emplace(alpha->evaluate(_it, _swl, _time).x);
    }
    return o;
}

luisa::optional<Bool> Surface::Closure::dispersive() const noexcept {
    return luisa::nullopt;
}

Surface::Evaluation Surface::Closure::evaluate(
    Expr<float3> wo, Expr<float3> wi,
    TransportMode mode) const noexcept {
    return _evaluate(wo, wi, mode);
}

Surface::Sample Surface::Closure::sample(
    Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
    TransportMode mode) const noexcept {
    return _sample(wo, u_lobe, u, mode);
}

void Surface::Closure::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df,
    TransportMode mode) const noexcept {
    _backward(wo, wi, df, mode);
}

}// namespace luisa::render
