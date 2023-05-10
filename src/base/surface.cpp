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

void Surface::Instance::closure(PolymorphicCall<Closure> &call,
                                const Interaction &it, const SampledWavelengths &swl,
                                Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    auto cls = call.collect(closure_identifier(), [&] {
        return create_closure(swl, time);
    });
    populate_closure(cls, it, wo, eta_i);
}

luisa::string Surface::Instance::closure_identifier() const noexcept {
    return luisa::string{node()->impl_type()};
}

}// namespace luisa::render
