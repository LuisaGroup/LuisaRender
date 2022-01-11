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
              "emission", make_float3(desc->property_float("emission")))} {}

    [[nodiscard]] float power(const Shape *shape) const noexcept override { return /* TODO */ 0.0f; }
    [[nodiscard]] uint property_flags() const noexcept override { return 0u; }
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
    [[nodiscard]] Evaluation evaluate(const Pipeline &pipeline, const Interaction &it, Expr<float3> p_from) const noexcept override {
        using namespace luisa::compute;
        auto params = pipeline.buffer<DiffuseLightParams>(it.shape()->light_buffer_id()).read(0u);
        auto pdf_area = pipeline.buffer<float>(it.shape()->pdf_buffer_id()).read(it.triangle_id()) * (1.0f / it.triangle_area());
        auto cos_wo = dot(it.wo(), it.shading().n());
        auto front_face = cos_wo > 0.0f;
        auto Le = make_float3(17.0f, 12.0f, 4.0f) * 2.0f;
        auto pdf = distance_squared(it.p(), p_from) * pdf_area * (1.0f / cos_wo);
        return {.Le = Le, .pdf = ite(front_face, pdf, 0.0f)};
    }
    [[nodiscard]] Sample sample(const Pipeline &pipeline, Sampler::Instance &sampler, const Interaction &it_from, Expr<uint> light_inst_id) const noexcept override {
        auto [light_inst, light_to_world] = pipeline.instance(light_inst_id);
        auto alias_table_buffer_id = light_inst->alias_table_buffer_id();
        auto params = pipeline.buffer<DiffuseLightParams>(light_inst->light_buffer_id()).read(0u);
        auto triangle_id = sample_alias_table(
            pipeline.buffer<AliasEntry>(alias_table_buffer_id),
            params.triangle_count, sampler.generate_1d());
        auto triangle = pipeline.triangle(light_inst, triangle_id);
        auto light_to_world_normal = transpose(inverse(light_to_world));
        auto uvw = sample_uniform_triangle(sampler.generate_2d());
        auto [p, ng, area] = pipeline.surface_point(light_inst, light_to_world, triangle, uvw);
        auto [ns, tangent, uv] = pipeline.surface_point_attributes(light_inst, light_to_world_normal, triangle, uvw);
        Interaction it{light_inst_id, light_inst, triangle_id, area, p, normalize(it_from.p() - p), ng, uv, ns, tangent};
        auto eval = evaluate(pipeline, it, it_from.p());
        return {.eval = eval, .p_light = p, .shadow_ray = it_from.spawn_ray_to(p)};
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DiffuseLight)
