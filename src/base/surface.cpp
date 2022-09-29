//
// Created by Mike on 2021/12/14.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/interaction.h>
#include <base/pipeline.h>

namespace luisa::render {

Surface::Surface(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SURFACE} {}

luisa::unique_ptr<Surface::Instance> Surface::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    LUISA_ASSERT(!is_null(), "Building null Surface.");
    LUISA_ASSERT(!(is_transmissive() && is_thin()), "Surface cannot be both transmissive and thin.");
    return _build(pipeline, command_buffer);
}

Surface::Closure::Closure(
    const Surface::Instance *instance, Interaction it,
    const SampledWavelengths &swl, Expr<float> time) noexcept
    : _instance{instance}, _it{std::move(it)}, _swl{swl}, _time{time} {}

Surface::Evaluation Surface::Closure::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
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
    if (instance()->node()->is_differentiable()) { _backward(wo, wi, df, mode); }
}

luisa::optional<Float> Surface::Closure::_opacity() const noexcept { return luisa::nullopt; }
luisa::optional<Bool> Surface::Closure::_is_dispersive() const noexcept { return luisa::nullopt; }
luisa::optional<Float> Surface::Closure::_eta() const noexcept { return luisa::nullopt; }

luisa::optional<Float> Surface::Closure::opacity() const noexcept {
    // We do not allow transmissive surfaces to be non-opaque.
    return instance()->node()->is_transmissive() ? luisa::nullopt : _opacity();
}

luisa::optional<Float> Surface::Closure::eta() const noexcept {
    // We do not care about eta of non-transmissive surfaces.
    if (instance()->node()->is_transmissive()) {
        auto eta = _eta();
        LUISA_ASSERT(eta.has_value(), "Transmissive surface must have eta.");
        return eta;
    }
    return luisa::nullopt;
}

luisa::optional<Bool> Surface::Closure::is_dispersive() const noexcept {
    return _is_dispersive();
}

}// namespace luisa::render
