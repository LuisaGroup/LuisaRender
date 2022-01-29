//
// Created by Mike Smith on 2022/1/12.
//

#include <luisa-compute.h>
#include <base/light.h>
#include <base/pipeline.h>
#include <util/sampling.h>

namespace luisa::render {

struct alignas(16) FakePointLightParams {
    float3 rsp;
    float scale;
    float radius;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::FakePointLightParams,
    rsp, scale, radius){};

namespace luisa::render {

class FakePointLight final : public Light {

private:
    FakePointLightParams _params{};

public:
    FakePointLight(Scene *scene, const SceneNodeDesc *desc) noexcept : Light{scene, desc} {
        auto emission = desc->property_float3_or_default(
            "emission", lazy_construct([desc] {
                return make_float3(desc->property_float("emission"));
            }));
        auto scale = desc->property_float_or_default("scale", 1.0f);
        std::tie(_params.rsp, _params.scale) = RGB2SpectrumTable::srgb().decode_unbound(
            max(emission * scale, 0.0f));
        _params.radius = desc->property_float_or_default("radius", 0.0f);
    }
    [[nodiscard]] bool is_black() const noexcept override { return _params.scale == 0.0f; }
    [[nodiscard]] bool is_virtual() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "fakepoint"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *shape) const noexcept override {
        if (!shape->is_virtual()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Virtual lights should not be "
                "applied to non-virtual shapes.");
        }
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<FakePointLightParams>(1u);
        command_buffer << buffer_view.copy_from(&_params);
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class FakePointLightClosure final : public Light::Closure {

private:
    const Pipeline &_pipeline;
    const SampledWavelengths &_swl;

public:
    explicit FakePointLightClosure(const Pipeline &ppl, const SampledWavelengths &swl) noexcept
        : _pipeline{ppl}, _swl{swl} {}
    [[nodiscard]] Light::Evaluation evaluate(const Interaction &, Expr<float3>) const noexcept override { return {}; /* should never be called */ }
    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, Expr<uint> light_inst_id, const Interaction &it_from) const noexcept override {
        using namespace luisa::compute;
        auto [inst, inst_to_world] = _pipeline.instance(light_inst_id);
        auto params = _pipeline.buffer<FakePointLightParams>(inst->light_buffer_id()).read(0u);
        RGBIlluminantSpectrum spec{
            RGBSigmoidPolynomial{params.rsp}, params.scale,
            DenselySampledSpectrum::cie_illum_d65()};
        auto L = spec.sample(_swl);
        auto center = make_float3(inst_to_world * make_float4(make_float3(0.0f), 1.0f));
        auto frame = Frame::make(normalize(it_from.p() - center));
        auto offset = sample_uniform_disk_concentric(sampler.generate_2d());
        auto p_light = params.radius * frame.local_to_world(make_float3(offset, 0.0f)) + center;
        Light::Sample s;
        static constexpr auto delta_pdf = 1e8f;
        s.eval.L = L * delta_pdf;
        s.eval.pdf = distance_squared(p_light, it_from.p()) * delta_pdf;
        s.shadow_ray = it_from.spawn_ray_to(p_light);
        return s;
    }
};

luisa::unique_ptr<Light::Closure> FakePointLight::decode(const Pipeline &pipeline, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<FakePointLightClosure>(pipeline, swl);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::FakePointLight)
