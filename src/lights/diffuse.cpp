//
// Created by Mike Smith on 2022/1/11.
//

#include <luisa-compute.h>
#include <scene/light.h>
#include <scene/interaction.h>

namespace luisa::render {

constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);

class DiffuseLight final : public Light {

public:
    DiffuseLight(Scene *scene, const SceneNodeDesc *desc) noexcept : Light{scene, desc} {}
    [[nodiscard]] float power(const Shape *shape) const noexcept override { return /* TODO */ 0.0f; }
    [[nodiscard]] uint property_flags() const noexcept override { return 0u; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "diffuse"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        return 0;
    }
    [[nodiscard]] Evaluation evaluate(const Interaction &it, Expr<float3> p_from) const noexcept override {
        auto light_area = length(cross(light_u, light_v));
        using namespace luisa::compute;
        auto cos_wo = dot(it.wo(), it.shading().n());
        auto front_face = cos_wo > 0.0f;
        auto Le = make_float3(17.0f, 12.0f, 4.0f) * 2.0f;
        auto pdf = distance_squared(it.p(), p_from) / (light_area * cos_wo);
        return {.Le = Le, .pdf = ite(front_face, pdf, 0.0f)};
    }
    [[nodiscard]] Sample sample(Sampler::Instance &sampler, Expr<float3> p_from, Expr<InstancedShape> light_inst, Expr<float4x4> light_inst_to_world) const noexcept override {
        auto u = sampler.generate_2d();
        auto p_light = u.x * light_u + u.y * light_v + light_position;
        auto L = p_light - p_from;
        auto wi = normalize(L);
        auto light_normal = normalize(cross(light_u, light_v));
        // TODO: two-sided
        Interaction it{light_inst, p_light, -wi, light_normal};
        auto eval = evaluate(it, p_from);
        return {.eval = eval, .p = p_light};
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DiffuseLight)
