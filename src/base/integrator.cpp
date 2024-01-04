//
// Created by Mike on 2021/12/14.
//

#include <base/scene.h>
#include <sdl/scene_node_desc.h>
#include <util/progress_bar.h>
#include <base/integrator.h>
#include <base/pipeline.h>
//#define LUISA_RENDER_PATH_REPLAY_DEBUG

namespace luisa::render {

Integrator::Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::INTEGRATOR},
      _sampler{scene->load_sampler(desc->property_node_or_default(
          "sampler", SceneNodeDesc::shared_default_sampler("independent")))},
      _light_sampler{scene->load_light_sampler(desc->property_node_or_default(
          "light_sampler", SceneNodeDesc::shared_default_light_sampler("uniform")))} {}

Integrator::Instance::Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept
    : _pipeline{pipeline}, _integrator{integrator},
      _sampler{integrator->sampler()->build(pipeline, command_buffer)},
      _light_sampler{pipeline.has_lighting() ?
                         integrator->light_sampler()->build(pipeline, command_buffer) :
                         nullptr} {}

ProgressiveIntegrator::Instance::Instance(Pipeline &pipeline,
                                          CommandBuffer &command_buffer,
                                          const ProgressiveIntegrator *node) noexcept
    : Integrator::Instance{pipeline, command_buffer, node} {}

ProgressiveIntegrator::Instance::~Instance() noexcept = default;

void ProgressiveIntegrator::Instance::render(Stream &stream) noexcept {
    CommandBuffer command_buffer{&stream};
    for (auto i = 0u; i < pipeline().camera_count(); i++) {
        auto camera = pipeline().camera(i);
        auto resolution = camera->film()->node()->resolution();
        auto pixel_count = resolution.x * resolution.y;
        camera->film()->prepare(command_buffer);
        _render_one_camera(command_buffer, camera);
        luisa::vector<float4> pixels(pixel_count);
        camera->film()->download(command_buffer, pixels.data());
        command_buffer << compute::synchronize();
        camera->film()->release();
        auto film_path = camera->node()->file();
        save_image(film_path, reinterpret_cast<const float *>(pixels.data()), resolution);
    }
}

void ProgressiveIntegrator::Instance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    auto pixel_count = resolution.x * resolution.y;
    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << pipeline().printer().reset();
    command_buffer << compute::synchronize();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel_id = dispatch_id().xy();
        auto L = Li(camera, frame_index, pixel_id, time);
        camera->film()->accumulate(pixel_id, shutter_weight * L);
    };

    Clock clock_compile;
    auto render = pipeline().device().compile(render_kernel);
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    Clock clock;
    ProgressBar progress;
    progress.update(0.);
    auto dispatch_count = 0u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline().update(command_buffer, s.point.time);
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            dispatch_count++;
            if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
            auto dispatches_per_commit = 4u;
            if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                dispatch_count = 0u;
                auto p = sample_id / static_cast<double>(spp);
                command_buffer << [&progress, p] { progress.update(p); };
            }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

Float3 ProgressiveIntegrator::Instance::Li(const Camera::Instance *camera, Expr<uint> frame_index,
                                           Expr<uint2> pixel_id, Expr<float> time) const noexcept {
    LUISA_ERROR_WITH_LOCATION("ProgressiveIntegrator::Li() is not implemented.");
}

ProgressiveIntegrator::ProgressiveIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : Integrator{scene, desc} {}

DifferentiableIntegrator::DifferentiableIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept
    : Integrator(scene, desc),
      _iterations{std::max(desc->property_uint_or_default("iterations", 100u), 1u)},
      _display_camera_index{desc->property_int_or_default("display_camera_index", -1)},
      _save_process{desc->property_bool_or_default("save_process", false)},
      _save_grad_map{desc->property_bool_or_default("save_grad_map", false)},
      _save_finite_diff{desc->property_bool_or_default("save_finite_diff", false)},
      _loss{scene->load_loss(desc->property_node_or_default(
          "loss", SceneNodeDesc::shared_default_loss("L2")))},
      _optimizer{scene->load_optimizer(desc->property_node_or_default(
          "optimizer", SceneNodeDesc::shared_default_optimizer("GD")))} {}

DifferentiableIntegrator::Instance::Instance(
    Pipeline &pipeline, CommandBuffer &command_buffer, const DifferentiableIntegrator *integrator) noexcept
    : Integrator::Instance{pipeline, command_buffer, integrator},
      _loss{node<DifferentiableIntegrator>()->loss()->build(pipeline, command_buffer)},
      _optimizer{node<DifferentiableIntegrator>()->optimizer()->build(pipeline, command_buffer)} {}

}// namespace luisa::render
