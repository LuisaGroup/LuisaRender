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
          "spectrum", SceneNodeDesc::shared_default_spectrum("srgb")))} {}

Integrator::Instance::Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept
    : _pipeline{pipeline}, _integrator{integrator},
      _sampler{integrator->sampler()->build(pipeline, command_buffer)},
      _light_sampler{pipeline.has_lighting() ?
                         integrator->light_sampler()->build(pipeline, command_buffer) :
                         nullptr},
      _spectrum{integrator->spectrum()->build(pipeline, command_buffer)} {}

DifferentiableIntegrator::DifferentiableIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : Integrator(scene, desc),
      _learning_rate{std::max(desc->property_float_or_default("learning_rate", 1.f), 0.f)},
      _iterations{std::max(desc->property_uint_or_default("iterations", 100u), 1u)},
      _display_camera_index{desc->property_int_or_default("display_camera_index", -1)},
      _save_process{desc->property_bool_or_default("save_process", false)},
      _loss_function{scene->load_loss(desc->property_node_or_default(
          "loss", SceneNodeDesc::shared_default_loss("L1")))} {

    // optimizer
    auto optimizer_str = desc->property_string_or_default("optimizer", "GD");
    for (auto &c : optimizer_str) { c = static_cast<char>(toupper(c)); }
    if (optimizer_str == "BGD") {
        _optimizer = Optimizer::BGD;
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Unsupported optimizer '{}'. "
            "Fallback to BGD optimizer.",
            optimizer_str);
        _optimizer = Optimizer::BGD;
    }
}

}// namespace luisa::render
