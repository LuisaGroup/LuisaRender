//
// Created by Mike Smith on 2022/1/11.
//

#include <luisa-compute.h>
#include <base/light.h>
#include <base/interaction.h>
#include <util/sampling.h>
#include <base/pipeline.h>

namespace luisa::render {

struct DiffuseLightParams {
    float3 rsp;
    float scale;
    uint triangle_count;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::DiffuseLightParams,
    rsp, scale, triangle_count){};

namespace luisa::render {

class DiffuseLight final : public Light {

private:
    float3 _rsp;
    float _scale{};

public:
    DiffuseLight(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Light{scene, desc} {
        auto emission = desc->property_float3_or_default(
            "emission", lazy_construct([desc] {
                return make_float3(desc->property_float("emission"));
            }));
        auto scale = desc->property_float_or_default("scale", 1.0f);
        std::tie(_rsp, _scale) = RGB2SpectrumTable::srgb().decode_unbound(
            max(emission * scale, 0.0f));
    }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f; }
    [[nodiscard]] bool is_virtual() const noexcept override { return false; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "diffuse"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        DiffuseLightParams params{
            .rsp = _rsp, .scale = _scale, .triangle_count = static_cast<uint>(shape->triangles().size())};
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<DiffuseLightParams>(1u);
        command_buffer << buffer_view.copy_from(&params) << luisa::compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class DiffuseLightClosure final : public Light::Closure {

private:
    const Pipeline &_pipeline;
    const SampledWavelengths &_swl;

private:
    [[nodiscard]] auto _evaluate(const Interaction &it_light, Expr<float3> p_from, const Var<DiffuseLightParams> &params) const noexcept {
        using namespace luisa::compute;
        auto pdf_triangle = _pipeline.buffer<float>(it_light.shape()->pdf_buffer_id()).read(it_light.triangle_id());
        auto pdf_area = cast<float>(params.triangle_count) * (pdf_triangle / it_light.triangle_area());
        auto cos_wo = dot(it_light.wo(), it_light.shading().n());
        auto front_face = cos_wo > 0.0f;
        RGBIlluminantSpectrum spec{
            RGBSigmoidPolynomial{params.rsp}, params.scale,
            DenselySampledSpectrum::cie_illum_d6500()};
        auto L = spec.sample(_swl);
        auto pdf = distance_squared(it_light.p(), p_from) * pdf_area * (1.0f / cos_wo);
        return Light::Evaluation{.L = L, .pdf = ite(front_face, pdf, 0.0f)};
    }

public:
    explicit DiffuseLightClosure(const Pipeline &pipeline, const SampledWavelengths &swl) noexcept
        : _pipeline{pipeline}, _swl{swl} {}
    [[nodiscard]] Light::Evaluation evaluate(const Interaction &it, Expr<float3> p_from) const noexcept override {
        auto params = _pipeline.buffer<DiffuseLightParams>(it.shape()->light_buffer_id()).read(0u);
        return _evaluate(it, p_from, params);
    }
    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, Expr<uint> light_inst_id, const Interaction &it_from) const noexcept override {
        using namespace luisa::compute;
        auto [light_inst, light_to_world] = _pipeline.instance(light_inst_id);
        auto params = _pipeline.buffer<DiffuseLightParams>(light_inst->light_buffer_id()).read(0u);
        auto alias_table_buffer_id = light_inst->alias_table_buffer_id();
        auto [triangle_id, _] = sample_alias_table(
            _pipeline.buffer<AliasEntry>(alias_table_buffer_id),
            params.triangle_count, sampler.generate_1d());
        auto triangle = _pipeline.triangle(light_inst, triangle_id);
        auto light_to_world_normal = transpose(inverse(light_to_world));
        auto uvw = sample_uniform_triangle(sampler.generate_2d());
        auto [p, ng, area] = _pipeline.surface_point_geometry(light_inst, light_to_world, triangle, uvw);
        auto [ns, tangent, uv] = _pipeline.surface_point_attributes(light_inst, light_to_world_normal, triangle, uvw);
        Interaction it_light{light_inst, triangle_id, area, p, normalize(it_from.p() - p), ng, uv, ns, tangent};
        DiffuseLightClosure closure{_pipeline, _swl};
        return {.eval = closure._evaluate(it_light, it_from.p(), params),
                .shadow_ray = it_from.spawn_ray_to(p)};
    }
};

luisa::unique_ptr<Light::Closure> DiffuseLight::decode(const Pipeline &pipeline, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<DiffuseLightClosure>(pipeline, swl);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DiffuseLight)
