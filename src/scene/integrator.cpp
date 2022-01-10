//
// Created by Mike on 2021/12/14.
//

#include <scene/scene.h>
#include <scene/sampler.h>
#include <sdl/scene_node_desc.h>
#include <scene/integrator.h>

namespace luisa::render {

[[nodiscard]] static auto default_light_distribution_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__default_light_distribution__", SceneNodeTag::LIGHT_DISTRIBUTION};
        d.define(SceneNodeTag::LIGHT_DISTRIBUTION, "Uniform", {});
        return &d;
    }();
    return desc;
}

[[nodiscard]] static auto default_sampler_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__default_sampler__", SceneNodeTag::SAMPLER};
        d.define(SceneNodeTag::SAMPLER, "Independent", {});
        return &d;
    }();
    return desc;
}

Integrator::Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node_or_default(
          "sampler", default_sampler_node_desc()))},
      _light_dist{scene->load_light_distribution(desc->property_node_or_default(
          "light_distribution", default_light_distribution_node_desc()))} {}

}// namespace luisa::render
