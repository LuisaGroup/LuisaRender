//
// Created by Mike Smith on 2022/1/11.
//

#include <base/light.h>
#include <base/interaction.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class DiffuseLight final : public Light {

private:
    const Texture *_emission;
    float _scale;
    bool _two_sided;

public:
    DiffuseLight(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Light{scene, desc},
          _emission{scene->load_texture(desc->property_node_or_default(
              "emission", SceneNodeDesc::shared_default_texture("Constant")))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)},
          _two_sided{desc->property_bool_or_default("two_sided", false)} {}
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto two_sided() const noexcept { return _two_sided; }
    [[nodiscard]] bool is_null() const noexcept override { return _scale == 0.0f || _emission->is_black(); }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class DiffuseLightInstance final : public Light::Instance {

private:
    const Texture::Instance *_texture;

public:
    DiffuseLightInstance(
        const Pipeline &ppl, const Light *light,
        const Texture::Instance *texture) noexcept
        : Light::Instance{ppl, light}, _texture{texture} {}
    [[nodiscard]] auto texture() const noexcept { return _texture; }
    [[nodiscard]] luisa::unique_ptr<Light::Closure> closure(
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Light::Instance> DiffuseLight::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto texture = pipeline.build_texture(command_buffer, _emission);
    return luisa::make_unique<DiffuseLightInstance>(pipeline, this, texture);
}

using namespace luisa::compute;

struct DiffuseLightClosure final : public Light::Closure {

    DiffuseLightClosure(const DiffuseLightInstance *light, const SampledWavelengths &swl, Expr<float> time) noexcept
        : Light::Closure{light, swl, time} {}

    [[nodiscard]] Light::Evaluation evaluate(const Interaction &it_light, Expr<float3> p_from) const noexcept override {
        using namespace luisa::compute;
        auto light = instance<DiffuseLightInstance>();
        auto &&pipeline = light->pipeline();
        auto pdf_triangle = pipeline.buffer<float>(it_light.shape()->pdf_buffer_id()).read(it_light.triangle_id());
        auto pdf_area = pdf_triangle / it_light.triangle_area();
        auto cos_wo = abs_dot(normalize(p_from - it_light.p()), it_light.ng());
        auto L = light->texture()->evaluate_illuminant_spectrum(it_light, _swl, _time).value *
                 light->node<DiffuseLight>()->scale();
        auto pdf = distance_squared(it_light.p(), p_from) * pdf_area * (1.0f / cos_wo);
        auto two_sided = light->node<DiffuseLight>()->two_sided();
        return {.L = ite(!two_sided & it_light.back_facing(), 0.f, L),
                .pdf = ite(!two_sided & it_light.back_facing(), 0.0f, pdf)};
    }

    [[nodiscard]] Light::Sample sample(Expr<uint> light_inst_id, const Interaction &it_from, Expr<float2> u_in) const noexcept override {
        auto light = instance<DiffuseLightInstance>();
        auto &&pipeline = light->pipeline();
        auto light_inst = pipeline.geometry()->instance(light_inst_id);
        auto light_to_world = pipeline.geometry()->instance_to_world(light_inst_id);
        auto alias_table_buffer_id = light_inst->alias_table_buffer_id();
        auto [triangle_id, ux] = sample_alias_table(
            pipeline.buffer<AliasEntry>(alias_table_buffer_id),
            light_inst->triangle_count(), u_in.x);
        auto triangle = pipeline.geometry()->triangle(light_inst, triangle_id);
        auto light_to_world_normal = transpose(inverse(make_float3x3(light_to_world)));
        auto uvw = sample_uniform_triangle(make_float2(ux, u_in.y));
        auto attrib = pipeline.geometry()->shading_point(
            light_inst, triangle, uvw, light_to_world, light_to_world_normal);
        auto light_wo = normalize(it_from.p() - attrib.pg);
        Interaction it_light{light_inst, light_inst_id, triangle_id, attrib, dot(light_wo, attrib.ng) < 0.0f};
        DiffuseLightClosure closure{light, _swl, _time};
        auto p_light = it_light.p_robust(light_wo);
        auto p_from = it_from.p_robust(-light_wo);
        return {.eval = closure.evaluate(it_light, it_from.p_shading()),
                .wi = -light_wo,
                .distance = distance(p_light, p_from) * .999f};
    }
};

luisa::unique_ptr<Light::Closure> DiffuseLightInstance::closure(
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<DiffuseLightClosure>(this, swl, time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DiffuseLight)
