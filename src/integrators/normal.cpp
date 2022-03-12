//
// Created by Mike on 2022/1/7.
//

#include <tinyexr.h>

#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

class NormalVisualizer;

class NormalVisualizerInstance final : public Integrator::Instance {

private:
    Pipeline &_pipeline;

private:
    static void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline,
        const Camera::Instance *camera, const Filter::Instance *filter,
        Film::Instance *film) noexcept;

public:
    explicit NormalVisualizerInstance(const NormalVisualizer *integrator, Pipeline &pipeline) noexcept;
    void render(Stream &stream) noexcept override;
};

class NormalVisualizer final : public Integrator {

public:
    NormalVisualizer(Scene *scene, const SceneNodeDesc *desc) noexcept : Integrator{scene, desc} {}
    [[nodiscard]] bool differentiable() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &) const noexcept override {
        return luisa::make_unique<NormalVisualizerInstance>(this, pipeline);
    }
};

NormalVisualizerInstance::NormalVisualizerInstance(
    const NormalVisualizer *integrator, Pipeline &pipeline) noexcept
    : Integrator::Instance{pipeline, integrator}, _pipeline{pipeline} {}

void NormalVisualizerInstance::render(Stream &stream) noexcept {
    luisa::vector<float> pixels;
    auto command_buffer = stream.command_buffer();
    for (auto i = 0u; i < _pipeline.camera_count(); i++) {
        auto [camera, film, filter] = _pipeline.camera(i);
        auto resolution = film->node()->resolution();
        auto pixel_count = resolution.x * resolution.y;
        _render_one_camera(command_buffer, _pipeline, camera, filter, film);
        pixels.resize(next_pow2(pixel_count) * 4u);
        film->download(command_buffer, reinterpret_cast<float4 *>(pixels.data()));
        command_buffer << compute::synchronize();
        auto film_path = camera->node()->file();
        if (film_path.extension() != ".exr") [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Unexpected film file extension. "
                "Changing to '.exr'.");
            film_path.replace_extension(".exr");
        }
        auto size = make_int2(resolution);
        const char *err = nullptr;
        SaveEXR(pixels.data(), size.x, size.y, 4, false,
                film_path.string().c_str(), &err);
        if (err != nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to save film to '{}'.",
                film_path.string());
        }
    }
}

void NormalVisualizerInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline,
    const Camera::Instance *camera, const Filter::Instance *filter, Film::Instance *film) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = film->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto sampler = pipeline.sampler();
    auto pixel_count = resolution.x * resolution.y;
    film->clear(command_buffer);
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    auto render = pipeline.device().compile<2>([&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal,
                                                   Float time, Float shutter_weight) noexcept {
        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id) + 0.5f;
        auto path_weight = def(1.f);
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        path_weight *= filter_weight;
        auto [ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        path_weight *= camera_weight;
        ray->set_origin(make_float3(camera_to_world * make_float4(ray->origin(), 1.0f)));
        ray->set_direction(normalize(camera_to_world_normal * ray->direction()));
        auto interaction = pipeline.intersect(ray);
        auto color = ite(
            interaction->valid(),
            interaction->shading().n() * 0.5f + 0.5f,
            make_float3());
        film->accumulate(pixel_id, shutter_weight * path_weight * color);
    });
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();
    Clock clock;
    auto sample_id = 0u;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 64u;
    for (auto s : shutter_samples) {
        if (pipeline.update_geometry(command_buffer, s.point.time)) { dispatch_count = 0u; }
        auto camera_to_world = camera->node()->transform()->matrix(s.point.time);
        auto camera_to_world_normal = transpose(inverse(make_float3x3(camera_to_world)));
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, camera_to_world, camera_to_world_normal,
                                     s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }
    command_buffer << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalVisualizer)
