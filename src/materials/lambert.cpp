//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <base/material.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

[[nodiscard]] static auto default_color_texture_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{
            "__lambert_material_default_color_texture",
            SceneNodeTag::TEXTURE};
        d.define(SceneNodeTag::TEXTURE, "const", {});
        return &d;
    }();
    return desc;
}

using namespace luisa::compute;

class LambertMaterial final : public Material {

private:
    const Texture *_color;

public:
    LambertMaterial(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Material{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", default_color_texture_desc()))} {
        if (!_color->is_color()) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in Lambert materials. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return "lambert"; }
    [[nodiscard]] bool is_black() const noexcept override { return _color->is_black(); }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<TextureHandle>(1u);
        auto texture = pipeline.encode_texture(command_buffer, _color);
        command_buffer << buffer_view.copy_from(texture);
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class LambertClosure final : public Material::Closure {

private:
    const Interaction &_interaction;
    Float4 _f;
    Float _cos_wo;
    Bool _front_face;

public:
    LambertClosure(const Interaction &it, Expr<float4> albedo) noexcept
        : _interaction{it},
          _f{albedo * inv_pi},
          _cos_wo{dot(it.wo(), it.shading().n())},
          _front_face{_cos_wo > 0.0f} {}

private:
    [[nodiscard]] Material::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto n = _interaction.shading().n();
        auto wo = _interaction.wo();
        auto cos_wi = dot(n, wi);
        auto same_hemisphere = cos_wi * _cos_wo > 0.0f;
        auto pdf = ite(same_hemisphere & _front_face, cosine_hemisphere_pdf(abs(cos_wi)), 0.0f);
        return {.f = _f, .pdf = std::move(pdf)};
    }

    [[nodiscard]] Material::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wi_local = sample_cosine_hemisphere(sampler.generate_2d());
        auto pdf = ite(_front_face, cosine_hemisphere_pdf(wi_local.z), 0.0f);
        wi_local.z *= sign(_cos_wo);
        return {.wi = _interaction.shading().local_to_world(wi_local),
                .eval = {.f = _f, .pdf = pdf}};
    }
};

luisa::unique_ptr<Material::Closure> LambertMaterial::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto texture = pipeline.buffer<TextureHandle>(it.shape()->material_buffer_id()).read(0u);
    auto R = pipeline.evaluate_texture(texture, it, swl, time);
    return luisa::make_unique<LambertClosure>(it, std::move(R));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LambertMaterial)
