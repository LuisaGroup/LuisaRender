//
// Created by Mike on 2022/1/7.
//

#include <scene/pipeline.h>
#include <scene/integrator.h>

namespace luisa::render {

class NormalVisualizer;

class NormalVisualizerInstance final : public Integrator::Instance {

private:
    static void _render_one_camera(
        Stream &stream, Pipeline &pipeline,
        const Camera::Instance *camera, const Filter::Instance *filter,
        Film::Instance *film) noexcept;

public:
    explicit NormalVisualizerInstance(const NormalVisualizer *integrator) noexcept;
    void render(Stream &stream, Pipeline &pipeline) noexcept override;
};

class NormalVisualizer final : public Integrator {

public:
    NormalVisualizer(Scene *scene, const SceneNodeDesc *desc) noexcept : Integrator{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "normal"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<NormalVisualizerInstance>(this);
    }
};

NormalVisualizerInstance::NormalVisualizerInstance(const NormalVisualizer *integrator) noexcept
    : Integrator::Instance{integrator} {}

void NormalVisualizerInstance::render(Stream &stream, Pipeline &pipeline) noexcept {
    for (auto i = 0u; i < pipeline.camera_count(); i++) {
        auto [camera, film, filter] = pipeline.camera(i);
        _render_one_camera(stream, pipeline, camera, filter, film);
        film->save(stream, camera->node()->file());
    }
}

void NormalVisualizerInstance::_render_one_camera(
    Stream &stream, Pipeline &pipeline,
    const Camera::Instance *camera,
    const Filter::Instance *filter,
    Film::Instance *film) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = film->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto sampler = pipeline.sampler();
    auto command_buffer = stream.command_buffer();
    film->clear(command_buffer);
    sampler->reset(command_buffer, resolution, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal, Float time) noexcept {
        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id);
        auto path_weight = def(make_float3(1.0f));
        if (filter == nullptr) {// not specified, using default box filter
            pixel += sampler->generate_2d();
        } else {
            auto [offset, filter_weight] = filter->sample(*sampler);
            pixel += offset;
            path_weight *= filter_weight;
        }
        auto [ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        sampler->save_state();
        path_weight *= camera_weight;

        if (camera->node()->transform() != nullptr) {
            ray.origin = make_float3(camera_to_world * make_float4(def<float3>(ray.origin), 1.0f));
            ray.direction = normalize(camera_to_world_normal * def<float3>(ray.direction));
        }
        auto interaction = pipeline.intersect(ray);
        auto color = ite(
            interaction->valid(),
            interaction->shading().n() * 0.5f + 0.5f,
            make_float3());
        film->accumulate(pixel_id, path_weight * color);
    };
    auto render = pipeline.device().compile(render_kernel);
    stream << synchronize();
    Clock clock;
    auto time_start = camera->node()->time_span().x;
    auto time_end = camera->node()->time_span().x;
    auto spp_per_commit = 16u;
    for (auto i = 0u; i < spp; i++) {
        auto t = static_cast<float>((static_cast<double>(i) + 0.5f) / static_cast<double>(spp));
        auto time = lerp(time_start, time_end, t);
        pipeline.update_geometry(command_buffer, time);
        auto camera_transform = camera->node()->transform();
        auto camera_to_world = camera_transform == nullptr ? make_float4x4() : camera_transform->matrix(t);
        auto camera_to_world_normal = transpose(inverse(make_float3x3(camera_to_world)));
        command_buffer << render(i, camera_to_world, camera_to_world_normal, time).dispatch(resolution);
        if (spp % spp_per_commit == spp_per_commit - 1u) [[unlikely]] { command_buffer << commit(); }
    }
    command_buffer << commit();
    stream << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalVisualizer)
