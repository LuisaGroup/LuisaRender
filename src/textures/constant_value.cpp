//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>

namespace luisa::render {

class ConstantValue final : public Texture {

private:
    float4 _v;

private:
    [[nodiscard]] TextureHandle encode(Pipeline &, CommandBuffer &) const noexcept override {
        return TextureHandle::encode_constant(handle_tag(), make_float3(), 0.0f, _v);
    }

public:
    ConstantValue(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        auto v = desc->property_float_list_or_default("v");
        if (v.size() > 4u) [[unlikely]] {
            LUISA_WARNING(
                "Too many values (count = {}) for ConstValue. "
                "Additional values will be discarded. [{}]",
                v.size(), desc->source_location().string());
            v.resize(4u);
        }
        for (auto i = 0u; i < v.size(); i++) {
            _v[i] = v[i];
        }
    }
    [[nodiscard]] bool is_black() const noexcept override { return all(_v == 0.0f); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constvalue"; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &, const Interaction &, const Var<TextureHandle> &handle,
        const SampledWavelengths &swl, Expr<float>) const noexcept override {
        return handle->extra();
    }
    [[nodiscard]] bool is_color() const noexcept override { return true; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantValue)
