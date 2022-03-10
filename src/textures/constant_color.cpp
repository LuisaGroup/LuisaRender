//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/interaction.h>
#include <base/pipeline.h>

namespace luisa::render {

class ConstantColor final : public Texture {

private:
    std::array<float, 3> _rsp{};
    bool _is_black{false};

public:
    ConstantColor(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        auto color = desc->property_float3_or_default(
            "color", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "color", 1.0f));
            }));
        auto rsp = RGB2SpectrumTable::srgb().decode_albedo(
            clamp(color, 0.0f, 1.0f));
        _rsp = {rsp.x, rsp.y, rsp.z};
        _is_black = all(color == 0.0f);
    }
    [[nodiscard]] Category category() const noexcept override { return Category::COLOR; }
    [[nodiscard]] auto rsp() const noexcept { return make_float3(_rsp[0], _rsp[1], _rsp[2]); }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return 3u; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantColorInstance final : public Texture::Instance {

private:
    luisa::optional<Differentiation::ConstantParameter> _diff_param;

public:
    ConstantColorInstance(
        const Pipeline &ppl, const Texture *texture,
        luisa::optional<Differentiation::ConstantParameter> param) noexcept
        : Texture::Instance{ppl, texture}, _diff_param{std::move(param)} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto rsp = [&] {
            if (_diff_param) {
                auto v = pipeline().differentiation().decode(*_diff_param);
                return RGBSigmoidPolynomial{v.xyz()};
            }
            return RGBSigmoidPolynomial{node<ConstantColor>()->rsp()};
        }();
        return RGBAlbedoSpectrum{rsp}.sample(swl);
    }
    void backward(const Interaction &it, const SampledWavelengths &swl, Expr<float> time, Expr<float4> grad) const noexcept override {
        if (_diff_param) {
            auto g = make_float4(
                dot(grad, sqr(swl.lambda())),
                dot(grad, swl.lambda()),
                dot(grad, make_float4(1.f)), 0.f);
            pipeline().differentiation().accumulate(*_diff_param, g);
        }
    }
};

luisa::unique_ptr<Texture::Instance> ConstantColor::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::optional<Differentiation::ConstantParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation().parameter(rsp()));
    }
    return luisa::make_unique<ConstantColorInstance>(
        pipeline, this, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantColor)
