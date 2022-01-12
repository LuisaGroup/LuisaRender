//
// Created by Mike Smith on 2022/1/11.
//

#include <luisa-compute.h>
#include <scene/light.h>
#include <scene/interaction.h>
#include <util/sampling.h>
#include <scene/pipeline.h>

namespace luisa::render {

struct DiffuseLightParams {
    float emission[3];
    uint triangle_count;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::DiffuseLightParams, emission, triangle_count){};

namespace luisa::render {

class DiffuseLight final : public Light {

private:
    float3 _emission;

public:
    DiffuseLight(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Light{scene, desc},
          _emission{desc->property_float3_or_default(
              "emission", make_float3(desc->property_float("emission")))} {
        _emission = max(_emission * desc->property_float_or_default("scale", 1.0f), 0.0f);
    }

    [[nodiscard]] float power(const Shape *shape) const noexcept override { return /* TODO */ 0.0f; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_emission == 0.0f); }
    [[nodiscard]] string_view impl_type() const noexcept override { return "diffuse"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        DiffuseLightParams params{};
        params.emission[0] = _emission.x;
        params.emission[1] = _emission.y;
        params.emission[2] = _emission.z;
        params.triangle_count = shape->triangles().size();
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<DiffuseLightParams>(1u);
        command_buffer << buffer_view.copy_from(&params) << luisa::compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const Interaction &it) const noexcept override;
};

class DiffuseLightClosure final : public Light::Closure {

private:
    const Pipeline &_pipeline;
    const Interaction &_it;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> p_from, const Var<DiffuseLightParams> &params) const noexcept {
        auto pdf_triangle = _pipeline.buffer<float>(_it.shape()->pdf_buffer_id()).read(_it.triangle_id());
        auto pdf_area = cast<float>(params.triangle_count) * (pdf_triangle / _it.triangle_area());
        auto cos_wo = dot(_it.wo(), _it.shading().n());
        auto front_face = cos_wo > 0.0f;
        auto emission = def<float3>(params.emission);
        auto pdf = distance_squared(_it.p(), p_from) * pdf_area * (1.0f / cos_wo);
        return Light::Evaluation{.Le = emission, .pdf = ite(front_face, pdf, 0.0f)};
    }

public:
    DiffuseLightClosure(const Pipeline &pipeline, const Interaction &it) noexcept
        : _pipeline{pipeline}, _it{it} {}

    [[nodiscard]] Light::Evaluation evaluate(Expr<float3> p_from) const noexcept override {
        using namespace luisa::compute;
        auto params = _pipeline.buffer<DiffuseLightParams>(_it.shape()->light_buffer_id()).read(0u);
        return _evaluate(p_from, params);
    }

    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, Expr<uint> light_inst_id) const noexcept override {
        auto [light_inst, light_to_world] = _pipeline.instance(light_inst_id);
        auto params = _pipeline.buffer<DiffuseLightParams>(light_inst->light_buffer_id()).read(0u);
        auto alias_table_buffer_id = light_inst->alias_table_buffer_id();
        auto triangle_id = sample_alias_table(
            _pipeline.buffer<AliasEntry>(alias_table_buffer_id),
            params.triangle_count, sampler.generate_1d());
        auto triangle = _pipeline.triangle(light_inst, triangle_id);
        auto light_to_world_normal = transpose(inverse(light_to_world));
        auto uvw = sample_uniform_triangle(sampler.generate_2d());
        auto [p, ng, area] = _pipeline.surface_point_geometry(light_inst, light_to_world, triangle, uvw);
        auto [ns, tangent, uv] = _pipeline.surface_point_attributes(light_inst, light_to_world_normal, triangle, uvw);
        Interaction it{light_inst_id, light_inst, triangle_id, area, p, normalize(_it.p() - p), ng, uv, ns, tangent};
        DiffuseLightClosure closure{_pipeline, it};
        return {.eval = closure._evaluate(_it.p(), params), .p_light = p, .shadow_ray = _it.spawn_ray_to(p)};
    }
};

luisa::unique_ptr<Light::Closure> DiffuseLight::decode(const Pipeline &pipeline, const Interaction &it) const noexcept {
    return luisa::make_unique<DiffuseLightClosure>(pipeline, it);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DiffuseLight)
