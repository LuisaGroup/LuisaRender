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
            // TODO: compute sampling distribution...
            for (auto i = 0u; i < image.resolution().x * image.resolution().y; i++) {
                auto &p = reinterpret_cast<float4 *>(image.pixels())[i];
                auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(p.xyz());
                p = make_float4(rsp, scale);
            }
            return image;
        });
    }
    [[nodiscard]] auto &image() const noexcept { return _image.get(); }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdri"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace luisa::compute;

class HDRIEnvironmentInstance final : public Environment::Instance {

private:
    uint _image_id{};

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, const SampledWavelengths &swl) const noexcept {
        auto theta = acos(wi_local.y);
        auto phi = atan2(wi_local.x, wi_local.z);
        auto u = -0.5f * inv_pi * phi;
        auto v = theta * inv_pi;
        auto s = static_cast<const HDRIEnvironment *>(node())->scale();
        auto rsp = pipeline().tex2d(_image_id).sample(make_float2(u, v));
        RGBIlluminantSpectrum spec{
            RGBSigmoidPolynomial{rsp.xyz()}, rsp.w * s,
            DenselySampledSpectrum::cie_illum_d6500()};
        auto L = spec.sample(swl);
        return Light::Evaluation{.L = L, .pdf = uniform_sphere_pdf()};
    }

public:
    HDRIEnvironmentInstance(Pipeline &pipeline, CommandBuffer &command_buffer, const HDRIEnvironment *env) noexcept
        : Environment::Instance{pipeline, env} {
        auto &&image = env->image();
        auto device_image = pipeline.create<Image<float>>(PixelStorage::FLOAT4, image.resolution());
        command_buffer << device_image->copy_from(image.pixels());
        _image_id = pipeline.register_bindless(*device_image, TextureSampler::bilinear_repeat());
    }
    // TODO: importance sampling
    [[nodiscard]] Light::Evaluation evaluate(
        Expr<float3> wi, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float>) const noexcept override {
        return _evaluate(transpose(env_to_world) * wi, swl);
    }
    [[nodiscard]] Light::Sample sample(
        Sampler::Instance &sampler, const Interaction &it_from, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float>) const noexcept override {
        auto wi = sample_uniform_sphere(sampler.generate_2d());
        return {.eval = _evaluate(wi, swl), .shadow_ray = it_from.spawn_ray(env_to_world * wi)};
    }
};

unique_ptr<Environment::Instance> HDRIEnvironment::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<HDRIEnvironmentInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HDRIEnvironment)
