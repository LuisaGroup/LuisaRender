//
// Created by ChenXin on 2022/2/24.
//

#include "grad_integrator.h"

namespace luisa::render {

GradIntegrator::GradIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::GRAD_INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node_or_default(
          "sampler", SceneNodeDesc::shared_default_sampler("Independent")))},
      _light_sampler{scene->load_light_sampler(desc->property_node_or_default(
          "light_sampler", SceneNodeDesc::shared_default_light_sampler("Uniform")))} {}

}