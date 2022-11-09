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
        _black = all(_v == 0.f);
    }
    [[nodiscard]] auto v() const noexcept { return _v; }
    [[nodiscard]] bool is_black() const noexcept override { return _black; }
    [[nodiscard]] bool is_constant() const noexcept override { return true; }
    [[nodiscard]] optional<float4> evaluate_static() const noexcept override { return _v; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantTextureInstance final : public Texture::Instance {

public:
    ConstantTextureInstance(const Pipeline &p, const Texture *t) noexcept
        : Texture::Instance{p, t} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  const SampledWavelengths &swl,
                                  Expr<float> time) const noexcept override {
        return node<ConstantTexture>()->v();
    }
};

luisa::unique_ptr<Texture::Instance> ConstantTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ConstantTextureInstance>(pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantTexture)
