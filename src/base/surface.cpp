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

//luisa::optional<Float> Surface::Closure::opacity() const noexcept {
//    // We do not allow transmissive surfaces to be non-opaque.
//    return instance()->node()->is_transmissive() ? luisa::nullopt : _opacity();
//}
//
//luisa::optional<Float> Surface::Closure::eta() const noexcept {
//    // We do not care about eta of non-transmissive surfaces.
//    if (instance()->node()->is_transmissive()) {
//        auto eta = _eta();
//        LUISA_ASSERT(eta.has_value(), "Transmissive surface must have eta.");
//        return eta;
//    }
//    return luisa::nullopt;
//}
//
//luisa::optional<Bool> Surface::Closure::is_dispersive() const noexcept {
//    if (instance()->pipeline().spectrum()->node()->is_fixed()) { return nullopt; }
//    return _is_dispersive();
//}

void Surface::Instance::closure(PolymorphicCall<Closure> &call,
                                const Interaction &it, const SampledWavelengths &swl,
                                Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    auto cls = call.collect(node()->closure_identifier(), [&] {
        return _create_closure(swl, time);
    });
    _populate_closure(cls, it, wo, eta_i);
}

}// namespace luisa::render
