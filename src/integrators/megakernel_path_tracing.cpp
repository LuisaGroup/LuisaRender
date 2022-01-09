//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/pipeline.h>
#include <scene/integrator.h>

namespace luisa::render {

class MegakernelPathTracing final : public Integrator {

private:
    uint _max_depth;

public:
    MegakernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 5u), 1u)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "megapath"; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegaPTInstance final : public Integrator::Instance {

private:
    uint _max_depth;

private:
    static void _render_one_camera(
        Stream &stream, Pipeline &pipeline,
        const Camera::Instance *camera,
        const Filter::Instance *filter,
        Film::Instance *film, uint max_depth) noexcept;

public:
    explicit MegaPTInstance(const MegakernelPathTracing *node) noexcept
        : Integrator::Instance{node}, _max_depth{node->max_depth()} {}
    void render(Stream &stream, Pipeline &pipeline) noexcept override {
        for (auto i = 0u; i < pipeline.camera_count(); i++) {
            auto [camera, film, filter] = pipeline.camera(i);
            _render_one_camera(stream, pipeline, camera, filter, film, _max_depth);
            film->save(stream, camera->node()->file());
        }
    }
};

unique_ptr<Integrator::Instance> MegakernelPathTracing::build(Pipeline &, CommandBuffer &) const noexcept {
    return luisa::make_unique<MegaPTInstance>(this);
}

void MegaPTInstance::_render_one_camera(
    Stream &stream, Pipeline &pipeline, const Camera::Instance *camera,
    const Filter::Instance *filter, Film::Instance *film, uint max_depth) noexcept {
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

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)
