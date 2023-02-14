#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/scene.h>

namespace luisa::render {

using namespace compute;

static const luisa::unordered_map<luisa::string_view, uint>
    aov_component_to_channels{{"sample", 3u},
                              {"diffuse", 3u},
                              {"specular", 3u},
                              {"normal", 3u},
                              {"albedo", 3u},
                              {"depth", 1u},
                              {"roughness", 2u},
                              {"ndc", 3u},
                              {"mask", 1u}};

class GradientPathTracing final : public Integrator {
private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    GradientPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GradientPathTracingInstance final : public Integrator::Instance {

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    luisa::optional<Window> _window;

private:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

public:
    explicit GradientPathTracingInstance(
        const GradientPathTracing *node,
        Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<GradientPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, camera);
            command_buffer << compute::synchronize();
            camera->film()->release();
        }
        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
    }
};

luisa::unique_ptr<Integrator::Instance> GradientPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<GradientPathTracingInstance>(
        this, pipeline, command_buffer);
}

void GradientPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {
    // auto spp = camera->node()->spp();
    // auto resolution = camera->film()->node()->resolution();
    // auto image_file = camera->node()->file();

    // auto pixel_count = resolution.x * resolution.y;
    // sampler()->reset(command_buffer, resolution, pixel_count, spp);
    // command_buffer << compute::synchronize();

    // LUISA_INFO(
    //     "Rendering to '{}' of resolution {}x{} at {}spp.",
    //     image_file.string(),
    //     resolution.x, resolution.y, spp);

    // using namespace luisa::compute;

    // Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
    //     set_block_size(16u, 16u, 1u);
    //     auto pixel_id = dispatch_id().xy();
    //     auto L = Li(camera, frame_index, pixel_id, time);
    //     camera->film()->accumulate(pixel_id, shutter_weight * L);
    // };

    // Clock clock_compile;
    // auto render = pipeline().device().compile(render_kernel);
    // auto integrator_shader_compilation_time = clock_compile.toc();
    // LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    // auto shutter_samples = camera->node()->shutter_samples();
    // command_buffer << synchronize();

    // LUISA_INFO("Rendering started.");
    // Clock clock;
    // ProgressBar progress;
    // progress.update(0.);
    // auto dispatch_count = 0u;
    // auto sample_id = 0u;
    // for (auto s : shutter_samples) {
    //     pipeline().update(command_buffer, s.point.time);
    //     for (auto i = 0u; i < s.spp; i++) {
    //         command_buffer << render(sample_id++, s.point.time, s.point.weight)
    //                               .dispatch(resolution);
    //         auto dispatches_per_commit =
    //             _display && !_display->should_close() ?
    //                 node<ProgressiveIntegrator>()->display_interval() :
    //                 32u;
    //         if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
    //             dispatch_count = 0u;
    //             auto p = sample_id / static_cast<double>(spp);
    //             if (_display && _display->update(command_buffer, sample_id)) {
    //                 progress.update(p);
    //             } else {
    //                 command_buffer << [&progress, p] { progress.update(p); };
    //             }
    //         }
    //     }
    // }
    // command_buffer << synchronize();
    // progress.done();

    // auto render_time = clock.toc();
    // LUISA_INFO("Rendering finished in {} ms.", render_time);    
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GradientPathTracing)
