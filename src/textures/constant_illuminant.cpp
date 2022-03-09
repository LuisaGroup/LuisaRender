//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class ConstantIlluminant final : public Texture {

private:
    float4 _rsp_scale;

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
        auto rsp_scale = RGB2SpectrumTable::srgb().decode_unbound(
            max(color, 0.0f) * max(scale, 0.0f));
        _rsp_scale = make_float4(rsp_scale.first, rsp_scale.second);
    }
    [[nodiscard]] auto rsp_scale() const noexcept { return _rsp_scale; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return _rsp_scale.w == 0.0f; }
    [[nodiscard]] uint channels() const noexcept override { return 4u; }
    [[nodiscard]] Category category() const noexcept override { return Category::ILLUMINANT; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantIlluminantInstance final : public Texture::Instance {

private:
    luisa::optional<Differentiation::ConstantParameter> _diff_param;

public:
    ConstantIlluminantInstance(
        const Pipeline &p, const Texture *t,
        luisa::optional<Differentiation::ConstantParameter> param) noexcept
        : Texture::Instance{p, t}, _diff_param{param} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto rsp_scale = [&] {
            return _diff_param ?
                       pipeline().differentiation().decode(*_diff_param) :
                       def(node<ConstantIlluminant>()->rsp_scale());
        }();
        auto rsp = RGBSigmoidPolynomial{rsp_scale.xyz()};
        auto spec = RGBIlluminantSpectrum{
            rsp, rsp_scale.w,
            DenselySampledSpectrum::cie_illum_d65()};
        return spec.sample(swl);
    }
    void backward(const Interaction &it, const SampledWavelengths &swl, Expr<float> time, Expr<float4> grad) const noexcept override {
        // TODO...
    }
};

luisa::unique_ptr<Texture::Instance> ConstantIlluminant::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::optional<Differentiation::ConstantParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation().parameter(_rsp_scale));
    }
    return luisa::make_unique<ConstantIlluminantInstance>(
        pipeline, this, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantIlluminant)
