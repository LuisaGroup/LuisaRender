//
// Created by Mike Smith on 2022/1/12.
//

#include <luisa-compute.h>
#include <scene/light.h>
#include <scene/pipeline.h>
#include <util/sampling.h>

namespace luisa::render {

struct alignas(16) FakePointLightParams {
    float emission[3];
    float radius;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::FakePointLightParams, emission, radius){};

namespace luisa::render {

class FakePointLight final : public Light {

private:
    FakePointLightParams _params{};

public:
    FakePointLight(Scene *scene, const SceneNodeDesc *desc) noexcept : Light{scene, desc} {
        auto emission = desc->property_float3_or_default("emission", [](auto desc) noexcept {
            return make_float3(desc->property_float("emission"));
        });
        auto scale = desc->property_float_or_default("scale", 1.0f);
        _params.emission[0] = std::max(emission.x * scale, 0.0f);
        _params.emission[1] = std::max(emission.y * scale, 0.0f);
        _params.emission[2] = std::max(emission.z * scale, 0.0f);
        _params.radius = desc->property_float_or_default("radius", 0.0f);
    }
    [[nodiscard]] bool is_black() const noexcept override {
        return _params.emission[0] == 0.0f &&
               _params.emission[1] == 0.0f &&
               _params.emission[2] == 0.0f;
    }
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
    [[nodiscard]] luisa::unique_ptr<Closure> decode(const Pipeline &pipeline) const noexcept override;
};

class FakePointLightClosure final : public Light::Closure {

private:
    const Pipeline &_pipeline;

public:
    explicit FakePointLightClosure(const Pipeline &ppl) noexcept: _pipeline{ppl} {}
    [[nodiscard]] Light::Evaluation evaluate(const Interaction &, Expr<float3>) const noexcept override { return {}; /* should never be called */ }
    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, Expr<uint> light_inst_id, const Interaction &it_from) const noexcept override {
        using namespace luisa::compute;
        auto [inst, inst_to_world] = _pipeline.instance(light_inst_id);
        auto params = _pipeline.buffer<FakePointLightParams>(inst->light_buffer_id()).read(0u);
        auto emission = def<float3>(params.emission);
        auto center = make_float3(inst_to_world * make_float4(make_float3(0.0f), 1.0f));
        auto offset = sample_uniform_sphere(sampler.generate_2d());
        auto p_light = params.radius * offset + center;
        Light::Sample s;
        s.eval.L = emission * 1e6f;
        s.eval.pdf = 1e6f * distance_squared(p_light, it_from.p());
        s.p_light = p_light;
        return s;
    }
};

luisa::unique_ptr<Light::Closure> FakePointLight::decode(const Pipeline &pipeline) const noexcept {
    return luisa::make_unique<FakePointLightClosure>(pipeline);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::FakePointLight)
