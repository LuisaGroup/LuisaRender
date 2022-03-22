//
// Created by Mike on 2021/12/14.
//

#include <base/scene.h>
#include <sdl/scene_node_desc.h>
#include <base/integrator.h>
#include <base/pipeline.h>

namespace luisa::render {

Integrator::Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node_or_default(
          "sampler", SceneNodeDesc::shared_default_sampler("independent")))},
      _light_sampler{scene->load_light_sampler(desc->property_node_or_default(
          "light_sampler", SceneNodeDesc::shared_default_light_sampler("uniform")))},
      _spectrum{scene->load_spectrum(desc->property_node_or_default(
          "spectrum", SceneNodeDesc::shared_default_spectrum("rgb")))} {}

Integrator::Instance::Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept
    : _pipeline{pipeline}, _integrator{integrator},
      _sampler{integrator->sampler()->build(pipeline, command_buffer)},
      _light_sampler{pipeline.has_lighting() ?
                         integrator->light_sampler()->build(pipeline, command_buffer) :
                         nullptr},
      _spectrum{integrator->spectrum()->build(pipeline, command_buffer)} {}

}// namespace luisa::render
