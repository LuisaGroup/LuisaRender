//
// Created by Mike Smith on 2022/1/10.
//

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <util/imageio.h>
#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class MegakernelPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return false; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPathTracingInstance final : public Integrator::Instance {

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    GLFWwindow *_window{nullptr};
    SwapChain _swapchain;
    Image<float> _image;
    luisa::vector<float4> _pixels;

private:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {

        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();

        static std::once_flag once_flag;
        std::call_once(once_flag, [] { glfwInit(); });

        if (_window != nullptr) {
            auto w = 0;
            auto h = 0;
            glfwGetWindowSize(_window, &w, &h);
            if (any(make_uint2(w, h) != resolution)) {
                glfwDestroyWindow(_window);
                _window = nullptr;
            }
        }

        if (_window == nullptr) {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, false);
            _window = glfwCreateWindow(resolution.x, resolution.y, "Mega-Kernel Path Tracer", nullptr, nullptr);
            auto window_handle = glfwGetWin32Window(_window);
            _swapchain = pipeline().device().create_swapchain(
                reinterpret_cast<uint64_t>(window_handle),
                command_buffer.stream(), resolution);
            _image = pipeline().device().create_image<float>(
                _swapchain.backend_storage(), resolution);
        }

        auto copy_shader = pipeline().device().compile<2>([&] {
            auto p = dispatch_id().xy();
            auto color = camera->film()->read(p).average;
            auto hdr2ldr = [](auto x) noexcept {
                return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                             12.92f * x,
                             x <= 0.00031308f),
                      0.0f, 1.0f);
            };
            _image.write(p, make_float4(hdr2ldr(color), 1.f));
        });

        camera->film()->clear(command_buffer);
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }

        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        command_buffer.commit();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;
        Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
            return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
        };

        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler()->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(*sampler(), pixel_id, time);
            auto swl = spectrum()->sample(*sampler());
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension()};

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);
            $for(depth, node<MegakernelPathTracing>()->max_depth()) {

                // trace
                auto it = pipeline().intersect(ray);

                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape()->has_light()) {
                        auto eval = light_sampler()->evaluate_hit(
                            *it, ray->origin(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    };
                }

                $if(!it->shape()->has_surface()) { $break; };

                // sample one light
                Light::Sample light_sample = light_sampler()->sample(
                    *sampler(), *it, swl, time);

                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().intersect_any(shadow_ray);

                // evaluate material
                SampledSpectrum eta_scale{swl.dimension(), 1.f};
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler()->generate_1d();
                auto u_bsdf = sampler()->generate_2d();
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    // apply roughness map
                    auto alpha_skip = def(false);
                    if (auto alpha_map = surface->alpha()) {
                        auto alpha = alpha_map->evaluate(*it, time).x;
                        alpha_skip = alpha < u_lobe;
                        u_lobe = ite(alpha_skip, (u_lobe - alpha) / (1.f - alpha), u_lobe / alpha);
                    }

                    $if(alpha_skip) {
                        ray = it->spawn_ray(ray->direction());
                        pdf_bsdf = 1e16f;
                    }
                    $else {
                        // create closure
                        auto closure = surface->closure(*it, swl, time);

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.wi;
                            auto eval = closure->evaluate(wi);
                            auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                            Li += mis_weight / light_sample.eval.pdf * abs(dot(eval.normal, wi)) *
                                  beta * eval.f * light_sample.eval.L;
                        };

                        // sample material
                        auto sample = closure->sample(u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                        beta *= abs(dot(sample.eval.normal, sample.wi)) * w * sample.eval.f;
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return isnan(b) | b <= 0.f; })) { $break; };
                auto rr_depth = node<MegakernelPathTracing>()->rr_depth();
                auto rr_threshold = node<MegakernelPathTracing>()->rr_threshold();
                auto q = swl.cie_y(beta);
                $if(depth >= rr_depth & q < 1.f) {
                    q = clamp(q, .05f, rr_threshold);
                    $if(sampler()->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };
            camera->film()->accumulate(pixel_id, swl.srgb(Li * shutter_weight));
        };
        auto render = pipeline().device().compile(render_kernel);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        ProgressBar progress;
        progress.update(0.0);

        auto sample_id = 0u;
        [&] {
            _framerate.clear();
            for (auto s : shutter_samples) {
                pipeline().update(command_buffer, s.point.time);
                for (auto i = 0u; i < s.spp; i++) {
                    command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                          .dispatch(resolution);
                    if (sample_id % 1u == 0u) [[unlikely]] {
                        auto p = sample_id / static_cast<double>(spp);
                        command_buffer << copy_shader().dispatch(resolution)
                                       << _swapchain.present(_image)
                                       << [&progress, p, this] {
                                              _framerate.record(1u);
                                              LUISA_INFO("{} spp/s", _framerate.report());
                                              progress.update(p);
                                          };
                    }
                    glfwPollEvents();
                    if (glfwWindowShouldClose(_window)) {
                        LUISA_INFO("Rendering aborted by user.");
                        return;
                    }
                }
            }
        }();
        command_buffer << synchronize();
        progress.done();
    }

public:
    explicit MegakernelPathTracingInstance(const MegakernelPathTracing *node, Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {}

    void render(Stream &stream) noexcept override {
        auto pt = node<MegakernelPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            _pixels.resize(next_pow2(pixel_count));
            _render_one_camera(command_buffer, camera);
            camera->film()->download(command_buffer, _pixels.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();
            save_image(film_path, (const float *)_pixels.data(), resolution);
        }
    }
};

unique_ptr<Integrator::Instance> MegakernelPathTracing::build(Pipeline &pipeline, CommandBuffer &cmd_buffer) const noexcept {
    return luisa::make_unique<MegakernelPathTracingInstance>(this, pipeline, cmd_buffer);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)
