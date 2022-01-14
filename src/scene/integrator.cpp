//
// Created by Mike on 2021/12/14.
//

#include <scene/scene.h>
#include <scene/sampler.h>
#include <sdl/scene_node_desc.h>
#include <scene/integrator.h>

namespace luisa::render {

[[nodiscard]] static auto default_light_sampler_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__integrator_default_light_sampler__", SceneNodeTag::LIGHT_SAMPLER};
        d.define(SceneNodeTag::LIGHT_SAMPLER, "Uniform", {});
        return &d;
    }();
    return desc;
}

[[nodiscard]] static auto default_sampler_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__integrator_default_sampler__", SceneNodeTag::SAMPLER};
        d.define(SceneNodeTag::SAMPLER, "Independent", {});
        return &d;
    }();
    return desc;
}

Integrator::Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node_or_default(
          "sampler", default_sampler_node_desc()))},
      _light_sampler{scene->load_light_sampler(desc->property_node_or_default(
          "light_sampler", default_light_sampler_node_desc()))} {}

}// namespace luisa::render
