//
// Created by Mike on 2021/12/14.
//

#include <scene/scene.h>
#include <scene/sampler.h>
#include <sdl/scene_node_desc.h>
#include <scene/integrator.h>

namespace luisa::render {

Integrator::Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node("sampler"))} {}

}// namespace luisa::render
