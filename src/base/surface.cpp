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
      _normal{scene->load_texture(desc->property_node_or_default("normal"))},
      _alpha{scene->load_texture(desc->property_node_or_default("alpha"))} {

    LUISA_RENDER_PARAM_CHANNEL_CHECK(Surface, normal, >=, 3);
    LUISA_RENDER_PARAM_CHANNEL_CHECK(Surface, alpha, ==, 1);
}

luisa::unique_ptr<Surface::Instance> Surface::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto instance = _build(pipeline, command_buffer);
    instance->_alpha = pipeline.build_texture(command_buffer, _alpha);
    instance->_normal = pipeline.build_texture(command_buffer, _normal);
    return instance;
}

Surface::Closure::Closure(
    const Surface::Instance *instance, Interaction it,
    const SampledWavelengths &swl, Expr<float> time) noexcept
    : _instance{instance}, _it{std::move(it)}, _swl{swl}, _time{time} {}

luisa::unique_ptr<Surface::Closure> Surface::Instance::closure(
    Interaction it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    if (_normal != nullptr) {
        auto normal_local = 2.f * _normal->evaluate(it, time).xyz() - 1.f;
        auto normal = it.shading().local_to_world(normal_local);
        it.set_shading(Frame::make(normal, it.shading().u()));
    }
    return _closure(std::move(it), swl, time);
}

}// namespace luisa::render
