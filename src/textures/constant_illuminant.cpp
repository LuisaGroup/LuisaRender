//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class ConstantIlluminant final : public Texture {

private:
    float4 _rsp_scale;

private:
    [[nodiscard]] TextureHandle _encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint handle_tag) const noexcept override {
        auto [buffer, buffer_id] = pipeline.arena_buffer<float4>(1u);
        command_buffer << buffer.copy_from(&_rsp_scale);
        return TextureHandle::encode_texture(handle_tag, buffer_id);
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
        auto rsp_scale = RGB2SpectrumTable::srgb().decode_unbound(
            max(color, 0.0f) * max(scale, 0.0f));
        _rsp_scale = make_float4(rsp_scale.first, rsp_scale.second);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constillum"; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &pipeline, const Interaction &,
        const Var<TextureHandle> &handle, Expr<float>) const noexcept override {
        return pipeline.buffer<float4>(handle->texture_id()).read(0u);
    }
    [[nodiscard]] bool is_black() const noexcept override { return _rsp_scale.w == 0.0f; }
    [[nodiscard]] Category category() const noexcept override { return Category::ILLUMINANT; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantIlluminant)
