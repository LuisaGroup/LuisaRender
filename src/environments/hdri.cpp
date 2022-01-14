//
// Created by Mike Smith on 2022/1/15.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <util/imageio.h>
#include <scene/interaction.h>
#include <scene/environment.h>
#include <scene/pipeline.h>

namespace luisa::render {

class HDRIEnvironment final : public Environment {

private:
    std::shared_future<LoadedImage<float>> _image;
    float _scale;

public:
    HDRIEnvironment(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)} {
        auto file_path = desc->property_path("file");
        _image = ThreadPool::global().async([file_path = std::move(file_path)] {
            Clock clock;
            auto image = load_hdr_image(file_path, 4u);
            LUISA_INFO(
                "Loaded HDRI image '{}' in {} ms.",
                file_path.string(), clock.toc());
            return image;
        });
    }
    [[nodiscard]] auto &image() const noexcept { return _image.get(); }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdri"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace luisa::compute;

class HDRIEnvironmentInstance final : public Environment::Instance {

private:
    uint _image_id{};
    float _scale;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, Expr<float> time) const noexcept {
        auto theta = acos(wi_local.y);
        auto phi = atan2(wi_local.x, wi_local.z);
        auto u = -0.5f * inv_pi * (phi + time);
        auto v = theta * inv_pi;
        auto L = pipeline().tex2d(_image_id).sample(make_float2(u, v));
        return Light::Evaluation{.L = make_float3(L) * _scale, .pdf = uniform_sphere_pdf()};
    }

public:
    HDRIEnvironmentInstance(Pipeline &pipeline, CommandBuffer &command_buffer, const HDRIEnvironment *env) noexcept
        : Environment::Instance{pipeline, env}, _scale{env->scale()} {
        auto &&image = env->image();
        auto device_image = pipeline.create<Image<float>>(PixelStorage::FLOAT4, image.resolution());
        command_buffer << device_image->copy_from(image.pixels());
        _image_id = pipeline.register_bindless(*device_image, TextureSampler::bilinear_repeat());
    }
    [[nodiscard]] Light::Evaluation evaluate(Expr<float3> wi, Expr<float3x3> env_to_world, Expr<float> time) const noexcept override {
        return _evaluate(transpose(env_to_world) * wi, time);
    }
    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, const Interaction &it_from, Expr<float3x3> env_to_world, Expr<float> time) const noexcept override {
        auto wi = sample_uniform_sphere(sampler.generate_2d());
        return {.eval = _evaluate(wi, time), .shadow_ray = it_from.spawn_ray(env_to_world * wi)};
    }
};

unique_ptr<Environment::Instance> HDRIEnvironment::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<HDRIEnvironmentInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HDRIEnvironment)
