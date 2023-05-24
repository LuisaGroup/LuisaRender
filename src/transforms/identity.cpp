//
// Created by Mike Smith on 2022/1/10.
//

#include <base/transform.h>

namespace luisa::render {

struct IdentityTransform final : public Transform {
    IdentityTransform(Scene *scene, const SceneNodeDesc *desc) noexcept : Transform{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_static() const noexcept override { return true; }
    [[nodiscard]] bool is_identity() const noexcept override { return true; }
    [[nodiscard]] float4x4 matrix(float time) const noexcept override { return make_float4x4(1.0f); }
};

}

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::IdentityTransform)
