//
// Created by Mike Smith on 2022/1/12.
//

#include <luisa-compute.h>
#include <base/material.h>
#include <base/interaction.h>
#include <base/pipeline.h>

namespace luisa::render {
struct FakeMirrorParams {
    float rsp[3];
    uint tex_id;
};
}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::FakeMirrorParams, rsp, tex_id) {
    [[nodiscard]] auto has_texture() const noexcept {
        return tex_id != ~0u;
    }
};
// clang-format on

namespace luisa::render {

class FakeMirrorMaterial final : public Material {

private:
    FakeMirrorParams _params{};
    bool _is_black{};

public:
    FakeMirrorMaterial(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Material{scene, desc} {
        auto color = clamp(
            desc->property_float3_or_default(
                "color", lazy_construct([desc] {
                    return make_float3(desc->property_float_or_default("color", 1.0f));
                })),
            0.0f, 1.0f);
        auto rsp = RGB2SpectrumTable::srgb().decode_albedo(color);
        _params.rsp[0] = rsp.x;
        _params.rsp[1] = rsp.y;
        _params.rsp[2] = rsp.z;
        _is_black = all(color == 0.0f);
    }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "fakemirror"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<FakeMirrorParams>(1u);
        command_buffer << buffer_view.copy_from(&_params);
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class FakeMirrorClosure final : public Material::Closure {

private:
    const Interaction &_it;
    Float4 _refl;

public:
    FakeMirrorClosure(const Interaction &it, Expr<float4> refl) noexcept
        : _it{it}, _refl{refl} {}
    [[nodiscard]] Material::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        return {.f = make_float4(0.0f), .pdf = 0.0f};
    }
    [[nodiscard]] Material::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto cos_wo = dot(_it.wo(), _it.shading().n());
        auto wi = 2.0f * cos_wo * _it.shading().n() - _it.wo();
        static constexpr auto delta_pdf = 1e8f;
        Material::Evaluation eval{
            .f = delta_pdf * _refl / cos_wo,
            .pdf = ite(cos_wo > 0.0f, delta_pdf, 0.0f)};
        return {.wi = std::move(wi), .eval = std::move(eval)};
    }
};

unique_ptr<Material::Closure> FakeMirrorMaterial::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<FakeMirrorParams>(it.shape()->material_buffer_id()).read(0u);
    using namespace luisa::compute;
    RGBSigmoidPolynomial rsp{def<float3>(params.rsp)};
    auto R = RGBAlbedoSpectrum{rsp}.sample(swl);
    $if (params->has_texture()) {

    };
    return luisa::make_unique<FakeMirrorClosure>(it, R);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::FakeMirrorMaterial)
