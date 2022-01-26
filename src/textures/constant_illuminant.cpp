//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>

namespace luisa::render {

class ConstantIlluminantTexture : public Texture {

private:
    float3 _rsp;
    float _scale{};

private:
    [[nodiscard]] TextureHandle encode(Pipeline &, CommandBuffer &) const noexcept override {
        return TextureHandle::encode_rsp_scale_constant(_rsp, _scale);
    }

public:
    ConstantIlluminantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        auto color = desc->property_float3_or_default(
            "emission", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "emission", 1.0f));
            }));
        std::tie(_rsp, _scale) = RGB2SpectrumTable::srgb().decode_unbound(max(color, 0.0f));
    }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constantilluminant"; }
    [[nodiscard]] uint handle_tag() const noexcept override { return TextureHandle::tag_rsp_scale_constant; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &, const Interaction &, const Var<TextureHandle> &handle,
        const SampledWavelengths &swl, Expr<float>) const noexcept override {
        RGBIlluminantSpectrum spec{
            handle->rsp(), handle->scale(),
            DenselySampledSpectrum::cie_illum_d6500()};
        return spec.sample(swl);
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantIlluminantTexture)
