//
// Created by Mike Smith on 2022/1/10.
//

#include <scene/transform.h>

namespace luisa::render {

class ScaleRotateTranslate final : public Transform {

private:
    float4x4 _matrix;

public:
    ScaleRotateTranslate(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Transform{scene, desc} {
        auto scaling = desc->property_float3_or_default("scale", [](auto desc) noexcept {
            return make_float3(desc->property_float_or_default("scale", 1.0f));
        });
        auto rotation = desc->property_float4_or_default("rotate", make_float4(0.0f, 0.0f, 1.0f, 0.0f));
        auto translation = desc->property_float3_or_default("translate", make_float3());
        _matrix = luisa::translation(translation) *
                  luisa::rotation(rotation.xyz(), radians(rotation.w)) *
                  luisa::scaling(scaling);
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return "srt"; }
    [[nodiscard]] bool is_static() const noexcept override { return true; }
    [[nodiscard]] float4x4 matrix(float) const noexcept override { return _matrix; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ScaleRotateTranslate)
