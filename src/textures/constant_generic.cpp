//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class ConstantValue final : public Texture {

private:
    float4 _v;

private:
    [[nodiscard]] TextureHandle _encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint handle_tag) const noexcept override {
        auto [buffer, buffer_id] = pipeline.arena_buffer<float4>(1u);
        command_buffer << buffer.copy_from(&_v);
        return TextureHandle::encode_texture(handle_tag, buffer_id);
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
        for (auto i = 0u; i < v.size(); i++) { _v[i] = v[i]; }
        LUISA_INFO("Generic v: [{}, {}, {}, {}].", _v.x, _v.y, _v.z, _v.w);
    }
    [[nodiscard]] bool is_black() const noexcept override { return all(_v == 0.0f); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &pipeline, const Interaction &,
        const Var<TextureHandle> &handle, Expr<float>) const noexcept override {
        return pipeline.buffer<float4>(handle->texture_id()).read(0u);
    }
    [[nodiscard]] Category category() const noexcept override { return Category::GENERIC; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantValue)
