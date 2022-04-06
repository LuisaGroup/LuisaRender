//
// Created by ChenXin on 2022/2/23.
//

#include <tinyexr.h>
#include <stb/stb_image_write.h>

#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <core/stl.h>
#include <stb/stb_image_write.h>

namespace luisa::render {

using namespace luisa::compute;

class MegakernelGradRadiative final : public Integrator {

public:
    enum Loss {
        L1 = 1,
        L2 = 2,
    };

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    Loss _loss_function;
    int _display_camera_index;
    uint _iterations;
    float _learning_rate;

public:
    MegakernelGradRadiative(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _display_camera_index{desc->property_int_or_default("display_camera_index", -1)},
          _iterations{std::max(desc->property_uint_or_default("iterations", 100u), 1u)},
          _learning_rate{std::max(desc->property_float_or_default("learning_rate", 1.f), 0.f)} {
        auto loss_str = desc->property_string_or_default("loss", "L2");
        for (auto &c : loss_str) { c = static_cast<char>(toupper(c)); }
        if (loss_str == "L1") {
            _loss_function = Loss::L1;
        } else if (loss_str == "L2") {
            _loss_function = Loss::L2;
        } else {
            LUISA_WARNING_WITH_LOCATION(
                "unknown loss '{}'. "
                "Fallback to L2 loss.",
                loss_str);
        }
    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto learning_rate() const noexcept { return _learning_rate; }
    [[nodiscard]] auto iterations() const noexcept { return _iterations; }
    [[nodiscard]] auto loss() const noexcept { return _loss_function; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return true; }
    [[nodiscard]] int display_camera_index() const noexcept { return _display_camera_index; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelGradRadiativeInstance final : public Integrator::Instance {

private:
    luisa::vector<float4> _pixels;
    luisa::optional<Window> _window;
    luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float4x4, float3x3, float, float>>
        bp_shaders, render_shaders;

private:
    void _render_one_camera(
        CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera,
        bool display = false) noexcept;

    void _integrate_one_camera(
        CommandBuffer &command_buffer, uint iteration, const Camera::Instance *camera) noexcept;
    static void _save_image(
        std::filesystem::path path,
        const luisa::vector<float4> &pixels, uint2 resolution) noexcept;

public:
    explicit MegakernelGradRadiativeInstance(
        const MegakernelGradRadiative *node,
        Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : Integrator::Instance{pipeline, command_buffer, node} {
        if (node->display_camera_index() >= 0) {
            LUISA_ASSERT(node->display_camera_index() < pipeline.camera_count(),
                         "display_camera_index exceeds camera count");

            auto film = pipeline.camera(node->display_camera_index())->film()->node();
            _window.emplace("Display", film->resolution(), true);
            auto pixel_count = film->resolution().x * film->resolution().y;
            _pixels.resize(next_pow2(pixel_count) * 4u);
        }

        std::filesystem::path output_dir{"outputs"};
        std::filesystem::remove_all(output_dir);
        std::filesystem::create_directories(output_dir);
    }

    void display(CommandBuffer &command_buffer, const Film::Instance *film, uint iteration) noexcept {
        static auto exposure = 0.f;
        static auto aces = false;
        static auto a = 2.51f;
        static auto b = 0.03f;
        static auto c = 2.43f;
        static auto d = 0.59f;
        static auto e = 0.14f;
        if (_window) {
            if (_window->should_close()) {
                _window.reset();
                return;
            }
            _window->run_one_frame([&] {
                auto resolution = film->node()->resolution();
                auto pixel_count = resolution.x * resolution.y;
                film->download(command_buffer, _pixels.data());
                command_buffer << synchronize();
                auto file_name = luisa::format("outputs/{:05}.exr", iteration);
                SaveEXR(reinterpret_cast<const float *>(_pixels.data()),
                        static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
                        false, file_name.c_str(), nullptr);
                auto scale = std::pow(2.f, exposure);
                auto pow = [](auto v, auto a) noexcept {
                    return make_float3(
                        std::pow(v.x, a),
                        std::pow(v.y, a),
                        std::pow(v.z, a));
                };
                auto tonemap = [](auto x) noexcept {
                    return x * (a * x + b) / (x * (c * x + d) + e);
                };
                for (auto &p : luisa::span{_pixels}.subspan(0u, pixel_count)) {
                    auto linear = scale * p.xyz();
                    if (aces) { linear = tonemap(linear); }
                    auto srgb = select(
                        1.055f * pow(linear, 1.0f / 2.4f) - 0.055f,
                        12.92f * linear,
                        linear <= 0.00304f);
                    p = make_float4(srgb, 1.f);
                }
                _window->set_background(_pixels.data(), resolution);
            });
        }
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<MegakernelGradRadiative>();
        auto command_buffer = stream.command_buffer();
        pipeline().printer().reset(stream);
        luisa::vector<float4> pixels;

        auto learning_rate = pt->learning_rate();
        auto iteration_num = pt->iterations();

        // delete output buffer
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto output_dir = camera->node()->file().parent_path() /
                              luisa::format("output_buffer_camera_{:03}", i);
            std::filesystem::remove_all(output_dir);
            std::filesystem::create_directories(output_dir);
        }

        for (auto k = 0u; k < iteration_num; ++k) {
            // render
            for (auto i = 0u; i < pipeline().camera_count(); i++) {
                auto camera = pipeline().camera(i);
                auto resolution = camera->film()->node()->resolution();
                auto pixel_count = resolution.x * resolution.y;
                auto output_path = camera->node()->file().parent_path() /
                                   luisa::format("output_buffer_camera_{:03}", i) /
                                   luisa::format("{:06}.exr", k);
                pixels.resize(next_pow2(pixel_count));

                // render
                _render_one_camera(command_buffer, k, camera, pt->display_camera_index() == i);

                // calculate grad
                _integrate_one_camera(command_buffer, k, camera);
            }
            // back propagate
            pipeline().differentiation().step(command_buffer, learning_rate);
        }

        // save results
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;

            _render_one_camera(command_buffer, iteration_num, camera);

            pixels.resize(next_pow2(pixel_count));
            camera->film()->download(command_buffer, pixels.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();

            _save_image(film_path, pixels, resolution);
        }
        std::cout << pipeline().printer().retrieve(stream);
        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
        pipeline().differentiation().dump(command_buffer, "outputs");
    }
};

unique_ptr<Integrator::Instance> MegakernelGradRadiative::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelGradRadiativeInstance>(this, pipeline, command_buffer);
}

void MegakernelGradRadiativeInstance::_integrate_one_camera(
    CommandBuffer &command_buffer, uint iteration, const Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->node()->film()->resolution();
    LUISA_INFO("Start backward propagation.");

    auto pt = this;

    auto sampler = pt->sampler();
    auto env = pipeline().environment();

    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();
    auto pt_exact = pt->node<MegakernelGradRadiative>();

    using namespace luisa::compute;

    auto shader_iter = bp_shaders.find(camera);
    if (shader_iter == bp_shaders.end()) {
        Kernel2D bp_kernel = [&](UInt frame_index, Float4x4 camera_to_world,
                                 Float3x3 env_to_world, Float time, Float shutter_weight) noexcept {
            set_block_size(8u, 8u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time, camera_to_world);
            auto swl = pt->spectrum()->sample(*sampler);
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension(), 1.0f};
            auto grad_weight = shutter_weight / Float(pixel_count * spp);

            auto it = Interaction{
                make_float3(1.0f),
                Float2{
                    (pixel_id.x + 0.5f) / resolution.x,
                    (pixel_id.y + 0.5f) / resolution.y}};

            // Loss
            Float3 loss;
            switch (pt_exact->loss()) {
                case MegakernelGradRadiative::Loss::L1:
                    // L1 loss
                    loss = ite(
                        camera->film()->read(pixel_id).average - camera->target()->evaluate(it, time).xyz() >= 0.0f,
                        1.0f,
                        -1.0f);
                    break;
                case MegakernelGradRadiative::Loss::L2:
                    // L2 loss
                    loss = 2.0f * (camera->film()->read(pixel_id).average -
                                   camera->target()->evaluate(it, time).xyz());
                    break;
            }
            for (auto i = 0u; i < 3u; ++i) {
                beta[i] *= loss[i];
            }

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);

            $for(depth, pt->node<MegakernelGradRadiative>()->max_depth()) {

                // trace
                auto it = pipeline().intersect(ray);

                // miss
                $if(!it->valid()) {
                    $break;
                };

                // hit light
                // TODO

                $if(!it->shape()->has_surface()) { $break; };

                // evaluate material
                SampledSpectrum eta_scale{swl.dimension(), 1.f};
                auto cos_theta_o = it->wo_local().z;
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                pipeline().dynamic_dispatch_surface(surface_tag, [&](auto surface) {
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

                        // sample material
                        auto sample = closure->sample(u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);

                        // radiative bp
                        closure->backward(sample.wi, beta * grad_weight * Li);

                        beta *= abs(dot(sample.eval.normal, sample.wi)) * w * sample.eval.f;
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(swl.cie_y(beta * eta_scale), .05f);
                auto rr_depth = pt->node<MegakernelGradRadiative>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelGradRadiative>()->rr_threshold();
                $if(depth >= rr_depth & q < rr_threshold) {
                    $if(sampler->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };
        };
        auto bp_shader = pipeline().device().compile(bp_kernel);
        shader_iter = bp_shaders.emplace(camera, std::move(bp_shader)).first;
    }
    auto &&bp_shader = shader_iter->second;
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 8u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline().update_geometry(command_buffer, s.point.time)) { dispatch_count = 0u; }
        auto camera_to_world = camera->node()->transform()->matrix(s.point.time);
        auto env_to_world = env == nullptr || env->node()->transform()->is_identity() ?
                                make_float3x3(1.0f) :
                                transpose(inverse(make_float3x3(
                                    env->node()->transform()->matrix(s.point.time))));
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << bp_shader(iteration * spp + sample_id++, camera_to_world,
                                        env_to_world, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }

    command_buffer << commit() << synchronize();
    LUISA_INFO("Backward propagation finished in {} ms.",
               clock.toc());
}

void MegakernelGradRadiativeInstance::_render_one_camera(
    CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera,
    bool display) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    camera->film()->clear(command_buffer);
    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto pt = this;
    auto light_sampler = pt->light_sampler();
    auto sampler = pt->sampler();
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    LUISA_INFO("");
    LUISA_INFO("Iteration = {}", iteration);
    LUISA_INFO(
        "Start rendering of resolution {}x{} at {}spp.",
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    auto shader_iter = render_shaders.find(camera);
    if (shader_iter == render_shaders.end()) {
        Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 env_to_world,
                                     Float time, Float shutter_weight) noexcept {
            set_block_size(8u, 8u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(
                *sampler, pixel_id, time, camera_to_world);
            auto swl = pt->spectrum()->sample(*sampler);
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension()};

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);
            $for(depth, pt->node<MegakernelGradRadiative>()->max_depth()) {

                // trace
                auto it = pipeline().intersect(ray);

                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(
                            ray->direction(), env_to_world, swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape()->has_light()) {
                        auto eval = light_sampler->evaluate_hit(
                            *it, ray->origin(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    };
                }

                $if(!it->shape()->has_surface()) { $break; };

                // sample one light
                Light::Sample light_sample = light_sampler->sample(
                    *sampler, *it, env_to_world, swl, time);

                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().intersect_any(shadow_ray);

                // evaluate material
                SampledSpectrum eta_scale{swl.dimension(), 1.f};
                auto cos_theta_o = it->wo_local().z;
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                pipeline().dynamic_dispatch_surface(surface_tag, [&](auto surface) {
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
                            Li += mis_weight / light_sample.eval.pdf *
                                  abs_dot(eval.normal, wi) *
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
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(swl.cie_y(beta * eta_scale), .05f);
                auto rr_depth = pt->node<MegakernelGradRadiative>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelGradRadiative>()->rr_threshold();
                $if(depth >= rr_depth & q < rr_threshold) {
                    $if(sampler->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };
            camera->film()->accumulate(pixel_id, swl.srgb(Li * shutter_weight));
        };
        auto render_shader = pipeline().device().compile(render_kernel);
        shader_iter = render_shaders.emplace(camera, std::move(render_shader)).first;
    }
    auto &&render_shader = shader_iter->second;
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 16u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline().update_geometry(command_buffer, s.point.time)) {
            dispatch_count = 0u;
        }
        auto camera_to_world = camera->node()->transform()->matrix(s.point.time);
        auto env_to_world = make_float3x3(1.f);
        if (auto env = pipeline().environment()) {
            env_to_world = transpose(inverse(make_float3x3(
                env->node()->transform()->matrix(s.point.time))));
        }
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render_shader(iteration * spp + sample_id++, camera_to_world, env_to_world,
                                            s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }
    command_buffer << synchronize();
    LUISA_INFO("Rendering finished in {} ms.",
               clock.toc());
    if (display) { pt->display(command_buffer, camera->film(), iteration); }
}
void MegakernelGradRadiativeInstance::_save_image(std::filesystem::path path,
                                                  const luisa::vector<float4> &pixels, uint2 resolution) noexcept {
    // save results
    auto pixel_count = resolution.x * resolution.y;
    auto size = make_int2(resolution);

    if (path.extension() != ".exr" && path.extension() != ".hdr") [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Unexpected film file extension. "
            "Changing to '.exr'.");
        path.replace_extension(".exr");
    }

    if (path.extension() == ".exr") {
        const char *err = nullptr;
        SaveEXR(reinterpret_cast<const float *>(pixels.data()),
                size.x, size.y, 4, false, path.string().c_str(), &err);
        if (err != nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to save film to '{}'.",
                path.string());
        }
    } else if (path.extension() == ".hdr") {
        stbi_write_hdr(path.string().c_str(), size.x, size.y, 4, reinterpret_cast<const float *>(pixels.data()));
    }
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelGradRadiative)
