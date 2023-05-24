//
// Created by Mike Smith on 2022/11/7.
//

#include <base/transform.h>

namespace luisa::render {

class ViewTransform final : public Transform {

private:
    float3 _origin;
    float3 _u;// right
    float3 _v;// up
    float3 _w;// back

public:
    ViewTransform(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Transform{scene, desc},
          _origin{desc->property_float3_or_default(
              "origin", lazy_construct([desc] {
                  return desc->property_float3_or_default("position");
              }))} {
        auto front = desc->property_float3_or_default("front", make_float3(0.0f, 0.0f, -1.0f));
        auto up = desc->property_float3_or_default("up", make_float3(0.0f, 1.0f, 0.0f));
        _w = normalize(-front);
        _u = normalize(cross(up, _w));
        _v = normalize(cross(_w, _u));
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_static() const noexcept override { return true; }
    [[nodiscard]] bool is_identity() const noexcept override {
        return all(_origin == 0.f) &&
               all(_v == make_float3(0.0f, 1.0f, 0.0f)) &&
               all(_w == make_float3(0.0f, 0.0f, 1.0f));
    }
    [[nodiscard]] float4x4 matrix(float time) const noexcept override {
        return make_float4x4(make_float4(_u, 0.f),
                             make_float4(_v, 0.f),
                             make_float4(_w, 0.f),
                             make_float4(_origin, 1.f));
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ViewTransform)
