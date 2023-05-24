//
// Created by Mike Smith on 2022/1/9.
//

#include <luisa-compute.h>
#include <base/shape.h>
#include <base/scene.h>

namespace luisa::render {

class ShapeInstance : public Shape {

private:
    const Shape *_shape;

public:
    ShapeInstance(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc}, _shape{scene->load_shape(desc->property_node("shape"))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] span<const Shape *const> children() const noexcept override { return {&_shape, 1u}; }
};

using InstanceWrapper = VisibilityShapeWrapper<ShapeInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::InstanceWrapper)
