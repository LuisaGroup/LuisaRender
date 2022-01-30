//
// Created by Mike Smith on 2022/1/10.
//

#include <base/transform.h>

namespace luisa::render {

class ScaleRotateTranslate final : public Transform {

private:
    float4x4 _matrix;

public:
    ScaleRotateTranslate(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Transform{scene, desc} {
        auto scaling = desc->property_float3_or_default("scale", lazy_construct([desc]{
            return make_float3(desc->property_float_or_default("scale", 1.0f));
        }));
        auto rotation = desc->property_float4_or_default("rotate", make_float4(0.0f, 0.0f, 1.0f, 0.0f));
        auto translation = desc->property_float3_or_default("translate", make_float3());
        _matrix = luisa::translation(translation) *
                  luisa::rotation(normalize(rotation.xyz()), radians(rotation.w)) *
                  luisa::scaling(scaling);
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_static() const noexcept override { return true; }
    [[nodiscard]] float4x4 matrix(float) const noexcept override { return _matrix; }
    [[nodiscard]] bool is_identity() const noexcept override {
        return all(_matrix[0] == make_float4(1.0f, 0.0f, 0.0f, 0.0f)) &&
               all(_matrix[1] == make_float4(0.0f, 1.0f, 0.0f, 0.0f)) &&
               all(_matrix[2] == make_float4(0.0f, 0.0f, 1.0f, 0.0f)) &&
               all(_matrix[3] == make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ScaleRotateTranslate)
