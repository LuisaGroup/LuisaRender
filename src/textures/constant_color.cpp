//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>

namespace luisa::render {

class ConstantColor final : public Texture {

private:
    std::array<float, 3> _rsp{};
    bool _is_black{false};

private:
    [[nodiscard]] TextureHandle _encode(Pipeline &, CommandBuffer &, uint handle_tag) const noexcept override {
        return TextureHandle::encode_constant(
            handle_tag, make_float3(_rsp[0], _rsp[1], _rsp[2]));
    }

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
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &, const Interaction &,
        const Var<TextureHandle> &handle, Expr<float>) const noexcept override {
        return compute::make_float4(handle->v(), 1.f);
    }
    [[nodiscard]] Category category() const noexcept override { return Category::COLOR; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantColor)
