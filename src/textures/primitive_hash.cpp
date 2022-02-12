//
// Created by Mike Smith on 2022/1/26.
//

#include <util/rng.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class PrimitiveHash final : public Texture {

private:
    uint _seed;

private:
    [[nodiscard]] TextureHandle _encode(Pipeline &, CommandBuffer &, uint handle_tag) const noexcept override {
        return TextureHandle::encode_constant(handle_tag, make_float3(luisa::bit_cast<float>(_seed)));
    }

public:
    PrimitiveHash(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _seed{desc->property_uint_or_default("seed", 19980810u)} {}
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &pipeline, const Interaction &it,
        const Var<TextureHandle> &handle, Expr<float>) const noexcept override {
        auto seed = as<uint>(handle->v().x);
        auto color = pcg3d(make_uint3(it.instance_id(), it.triangle_id(), seed));
        auto spec = pipeline.srgb_albedo_spectrum(
            make_float3(color) * static_cast<float>(1. / ~0u));
        return make_float4(spec.rsp().c(), 1.f);
    }
    [[nodiscard]] Category category() const noexcept override { return Category::COLOR; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PrimitiveHash)
