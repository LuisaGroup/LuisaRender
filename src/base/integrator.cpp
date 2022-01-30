//
// Created by Mike on 2021/12/14.
//

#include <base/scene.h>
#include <base/sampler.h>
#include <sdl/scene_node_desc.h>
#include <base/integrator.h>

namespace luisa::render {

Integrator::Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node_or_default(
          "sampler", SceneNodeDesc::shared_default_sampler("Independent")))},
      _light_sampler{scene->load_light_sampler(desc->property_node_or_default(
          "light_sampler", SceneNodeDesc::shared_default_light_sampler("Uniform")))} {}

}// namespace luisa::render
