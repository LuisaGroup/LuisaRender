//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>

namespace luisa::render {

class ConstantIlluminant final : public Texture {

private:
    float3 _rsp;
    float _scale{};

private:
    [[nodiscard]] TextureHandle encode(Pipeline &, CommandBuffer &) const noexcept override {
        return TextureHandle::encode_constant(
            handle_tag(), {}, 0.0f,
            make_float4(_rsp, _scale));
    }

public:
    ConstantIlluminant(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        auto color = desc->property_float3_or_default(
            "emission", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "emission", 1.0f));
            }));
        auto scale = desc->property_float3_or_default(
            "scale", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "scale", 1.0f));
            }));
        std::tie(_rsp, _scale) = RGB2SpectrumTable::srgb().decode_unbound(
            max(color, 0.0f) * max(scale, 0.0f));
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constillum"; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &, const Interaction &, const Var<TextureHandle> &handle,
        const SampledWavelengths &swl, Expr<float>) const noexcept override {
        auto rsp_scale = handle->extra();
        RGBIlluminantSpectrum spec{
            RGBSigmoidPolynomial{rsp_scale.xyz()}, rsp_scale.w,
            DenselySampledSpectrum::cie_illum_d6500()};
        return spec.sample(swl);
    }
    [[nodiscard]] bool is_color() const noexcept override { return false; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return true; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantIlluminant)
