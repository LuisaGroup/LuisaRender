//
// Created by Leon Kang on 2024/4/11.
//

//
// Created by Mike Smith on 2022/1/10.
//

#include <util/progress_bar.h>
#include <util/imageio.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class ReSTIRDirectLighting final : public ProgressiveIntegrator {

private:
    uint _num_nee_sample;
    uint _num_bsdf_sample;
    uint _num_neighbor_sample;
    /*
     * temporal reuse currently disabled for (offline) progressive rendering
     * bool _enable_temporal_reuse;
     */
    bool _enable_spatial_reuse;
    bool _enable_visibility_reuse;

public:
    ReSTIRDirectLighting(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _num_nee_sample{desc->property_uint_or_default("num_nee_sample", 1u)},
          _num_bsdf_sample{desc->property_uint_or_default("num_bsdf_sample", 1u)},
          _num_neighbor_sample{desc->property_uint_or_default("num_neighbor_sample", 5u)},
          _enable_spatial_reuse{desc->property_bool_or_default("enable_spatial_reuse", true)},
          _enable_visibility_reuse{desc->property_bool_or_default("enable_visibility_reuse", true)} {}
    [[nodiscard]] auto num_nee_sample() const noexcept { return _num_nee_sample; }
    [[nodiscard]] auto num_bsdf_sample() const noexcept { return _num_bsdf_sample; }
    [[nodiscard]] auto num_neighbor_sample() const noexcept { return _num_neighbor_sample; }
    [[nodiscard]] auto enable_spatial_reuse() const noexcept { return _enable_spatial_reuse; }
    [[nodiscard]] auto enable_visibility_reuse() const noexcept { return _enable_visibility_reuse; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ReSTIRDirectLightingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer,
                            Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
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
        luisa::vector<float4> local_pixels;
        if (node()->video()) {
            shutter_samples = camera->node()->uniform_shutter_samples();
            local_pixels.resize(pixel_count);
        }
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;

        auto shutter_id = 0u;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            for (auto i = 0u; i < s.spp; i++) {
                command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                      .dispatch(resolution);
                if (auto &&p = pipeline().printer(); !p.empty()) {
                    command_buffer << p.retrieve();
                }
                dispatch_count++;
                if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
                auto dispatches_per_commit = 4u;
                if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
            if (node()->video()) {
                command_buffer << synchronize();
                camera->film()->download(command_buffer, local_pixels.data());
                command_buffer << compute::synchronize();
                camera->film()->clear(command_buffer);
                auto film_path = camera->node()->file();
                //film_path is a std::filesystem::path, add number to its name
                auto new_name = film_path.stem().string() + std::format("{:05}", shutter_id) + film_path.extension().string();
                auto new_film_path = film_path.replace_filename(new_name);
                save_image(new_film_path, reinterpret_cast<const float *>(local_pixels.data()), resolution);
                shutter_id++;
            }
        }
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> ReSTIRDirectLighting::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ReSTIRDirectLightingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ReSTIRDirectLighting)