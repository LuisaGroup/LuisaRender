//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>
#include <util/rng.h>

namespace luisa::render {

class ConstantTexture final : public Texture {

private:
    float4 _v;
    bool _black{false};
    uint _channels{0u};

public:
    ConstantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        auto scale = desc->property_float_or_default("scale", 1.f);
        auto v = desc->property_float_list_or_default("v");
        if (v.empty()) [[unlikely]] {
            LUISA_WARNING(
                "No value for ConstantTexture. "
                "Fallback to single-channel zero. [{}]",
                desc->source_location().string());
            v.emplace_back(0.f);
        } else if (v.size() > 4u) [[unlikely]] {
            LUISA_WARNING(
                "Too many values (count = {}) for ConstantTexture. "
                "Additional values will be discarded. [{}]",
                v.size(), desc->source_location().string());
            v.resize(4u);
        }
        _channels = v.size();
        for (auto i = 0u; i < v.size(); i++) { _v[i] = scale * v[i]; }
        switch (semantic()) {
            case Semantic::ALBEDO: {
                if (_channels == 1u) {
                    _v[1] = _v[0];
                    _v[2] = _v[0];
                } else if (_channels == 2u) {
                    LUISA_ERROR_WITH_LOCATION(
                        "ConstantTexture with semantic 'albedo' "
                        "requires 1, 3, or 4 channels, but got 2.");
                }
                _black = all(_v.xyz() <= 0.f);
                _v = scene->spectrum()->encode_srgb_albedo(_v.xyz());
                LUISA_INFO("Encoded: ({}, {}, {}, {})", _v[0], _v[1], _v[2], _v[3]);
                _channels = compute::pixel_storage_channel_count(
                    scene->spectrum()->encoded_albedo_storage(PixelStorage::FLOAT4));
                break;
            }
            case Semantic::ILLUMINANT:
                if (_channels == 1u) {
                    _v[1] = _v[0];
                    _v[2] = _v[0];
                } else if (_channels == 2u) {
                    LUISA_ERROR_WITH_LOCATION(
                        "ConstantTexture with semantic 'illuminant' "
                        "requires 1, 3, or 4 channels, but got 2.");
                }
                _black = all(_v.xyz() <= 0.f);
                _v = scene->spectrum()->encode_srgb_illuminant(_v.xyz());
                _channels = compute::pixel_storage_channel_count(
                    scene->spectrum()->encoded_illuminant_storage(PixelStorage::FLOAT4));
                break;
            case Semantic::GENERIC:
                _black = all(_v == 0.f);
                break;
        }
    }
    [[nodiscard]] auto v() const noexcept { return _v; }
    [[nodiscard]] bool is_black() const noexcept override { return _black; }
    [[nodiscard]] bool is_constant() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantTextureInstance final : public Texture::Instance {

private:
    luisa::optional<Differentiation::ConstantParameter> _diff_param;

public:
    ConstantTextureInstance(
        const Pipeline &p, const Texture *t,
        luisa::optional<Differentiation::ConstantParameter> param) noexcept
        : Texture::Instance{p, t}, _diff_param{std::move(param)} {}
    [[nodiscard]] Float4 evaluate(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (_diff_param) { return pipeline().differentiation().decode(*_diff_param); }
        return def(node<ConstantTexture>()->v());
    }
    void backward(const Interaction &it, const SampledWavelengths &swl,
                  Expr<float>, Expr<float4> grad) const noexcept override {
        if (_diff_param) {
            auto slot_seed = xxhash32(as<uint3>(it.p()));
            pipeline().differentiation().accumulate(*_diff_param, grad, slot_seed);
        }
    }
};

luisa::unique_ptr<Texture::Instance> ConstantTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::optional<Differentiation::ConstantParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation().parameter(_v, _channels, range()));
    }
    return luisa::make_unique<ConstantTextureInstance>(pipeline, this, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantTexture)
