//
// Created by Mike Smith on 2022/1/12.
//

#include <luisa-compute.h>
#include <scene/material.h>
#include <scene/interaction.h>
#include <scene/pipeline.h>

namespace luisa::render {

class FakeMirrorMaterial final : public Material {

private:
    float3 _color;

public:
    FakeMirrorMaterial(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Material{scene, desc} {
        _color = desc->property_float3_or_default("color", [](auto desc) noexcept {
            return make_float3(desc->property_float_or_default("color", 1.0f));
        });
        _color = clamp(_color, 0.0f, 1.0f);
    }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "fakemirror"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<float3>(1u);
        command_buffer << buffer_view.copy_from(&_color);
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const Interaction &it) const noexcept override;
};

class FakeMirrorClosure final : public Material::Closure {

private:
    const Interaction &_it;
    Float3 _color;

public:
    FakeMirrorClosure(const Interaction &it, Expr<float3> color) noexcept
        : _it{it}, _color{color} {}
    [[nodiscard]] Material::Evaluation evaluate(Expr<float3> wi, Expr<float> time) const noexcept override {
        return {.f = make_float3(0.0f), .pdf = 0.0f};
    }
    [[nodiscard]] Material::Sample sample(Sampler::Instance &sampler, Expr<float> time) const noexcept override {
        auto cos_wo = dot(_it.wo(), _it.shading().n());
        auto wi = 2.0f * cos_wo * _it.shading().n() - _it.wo();
        static constexpr auto delta_pdf = 1e8f;
        Material::Evaluation eval{
            .f = delta_pdf * _color / cos_wo,
            .pdf = ite(cos_wo > 0.0f, delta_pdf, 0.0f)};
        return {.wi = std::move(wi), .eval = std::move(eval)};
    }
};

unique_ptr<Material::Closure> FakeMirrorMaterial::decode(const Pipeline &pipeline, const Interaction &it) const noexcept {
    auto color = pipeline.buffer<float3>(it.shape()->material_buffer_id()).read(0u);
    return luisa::make_unique<FakeMirrorClosure>(it, color);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::FakeMirrorMaterial)
