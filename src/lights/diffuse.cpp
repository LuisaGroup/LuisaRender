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

public:
    DiffuseLight(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Light{scene, desc},
          _emission{scene->load_texture(desc->property_node_or_default(
              "emission", SceneNodeDesc::shared_default_texture("ConstIllum")))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)} {
        if (_emission->category() != Texture::Category::ILLUMINANT) [[unlikely]] {
            LUISA_ERROR(
                "Non-illuminant textures are not "
                "allowed in Diffuse lights. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] bool is_null() const noexcept override { return _scale == 0.0f || _emission->is_black(); }
    [[nodiscard]] bool is_virtual() const noexcept override { return false; }
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

class DiffuseLightClosure final : public Light::Closure {

private:
    const DiffuseLightInstance *_light;
    const SampledWavelengths &_swl;
    Float _time;

public:
    explicit DiffuseLightClosure(const DiffuseLightInstance *light, const SampledWavelengths &swl, Expr<float> time) noexcept
        : _light{light}, _swl{swl}, _time{time} {}
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] Light::Evaluation evaluate(const Interaction &it_light, Expr<float3> p_from) const noexcept override {
        using namespace luisa::compute;
        auto &&pipeline = _light->pipeline();
        auto pdf_triangle = pipeline.buffer<float>(it_light.shape()->pdf_buffer_id()).read(it_light.triangle_id());
        auto pdf_area = cast<float>(it_light.shape()->triangle_count()) * (pdf_triangle / it_light.triangle_area());
        auto cos_wo = dot(it_light.wo(), it_light.shading().n());
        auto front_face = cos_wo > 0.0f;
        auto L = _light->texture()->evaluate(it_light, _swl, _time) *
                 _light->node<DiffuseLight>()->scale();
        auto pdf = distance_squared(it_light.p(), p_from) * pdf_area * (1.0f / cos_wo);
        return Light::Evaluation{.L = L, .pdf = ite(front_face, pdf, 0.0f)};
    }
    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, Expr<uint> light_inst_id, const Interaction &it_from) const noexcept override {
        auto &&pipeline = _light->pipeline();
        auto [light_inst, light_to_world] = pipeline.instance(light_inst_id);
        auto alias_table_buffer_id = light_inst->alias_table_buffer_id();
        auto [triangle_id, _] = sample_alias_table(
            pipeline.buffer<AliasEntry>(alias_table_buffer_id),
            light_inst->triangle_count(), sampler.generate_1d());
        auto triangle = pipeline.triangle(light_inst, triangle_id);
        auto light_to_world_normal = transpose(inverse(make_float3x3(light_to_world)));
        auto uvw = sample_uniform_triangle(sampler.generate_2d());
        auto [p, ng, area] = pipeline.surface_point_geometry(light_inst, light_to_world, triangle, uvw);
        auto [ns, tangent, uv] = pipeline.surface_point_attributes(light_inst, light_to_world_normal, triangle, uvw);
        Interaction it_light{
            light_inst, light_inst_id, triangle_id, area, p,
            normalize(it_from.p() - p), ng, uv, ns, tangent};
        DiffuseLightClosure closure{_light, _swl, _time};
        return {.eval = closure.evaluate(it_light, it_from.p()),
                .shadow_ray = it_from.spawn_ray_to(p)};
    }
};

luisa::unique_ptr<Light::Closure> DiffuseLightInstance::closure(
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<DiffuseLightClosure>(this, swl, time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DiffuseLight)
