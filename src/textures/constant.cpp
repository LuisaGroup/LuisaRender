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
    uint _channels{0u};
    bool _black{false};
    bool _should_inline;

public:
    ConstantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _should_inline{desc->property_bool_or_default("inline", true)} {
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
    [[nodiscard]] bool should_inline() const noexcept { return _should_inline; }
    [[nodiscard]] optional<float4> evaluate_static() const noexcept override {
        return _should_inline ? luisa::make_optional(_v) : luisa::nullopt;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantTextureInstance final : public Texture::Instance {

private:
    uint _constant_slot{};

public:
    ConstantTextureInstance(Pipeline &p,
                            const ConstantTexture *t,
                            CommandBuffer &cmd_buffer) noexcept
        : Texture::Instance{p, t} {
        if (!t->should_inline()) {
            auto [buffer, buffer_id] = p.allocate_constant_slot();
            auto v = t->v();
            cmd_buffer << buffer.copy_from(&v) << compute::commit();
            _constant_slot = buffer_id;
        }
    }
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  const SampledWavelengths &swl,
                                  Expr<float> time) const noexcept override {
        if (auto texture = node<ConstantTexture>();
            texture->should_inline()) { return texture->v(); }
        return pipeline().constant(_constant_slot);
    }
};

luisa::unique_ptr<Texture::Instance> ConstantTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ConstantTextureInstance>(
        pipeline, this, command_buffer);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantTexture)
