//
// Created by Mike Smith on 2022/1/9.
//

#include <luisa-compute.h>
#include <base/shape.h>
#include <base/scene.h>

namespace luisa::render {

class ShapeInstance final : public Shape {

private:
    const Shape *_shape;

public:
    ShapeInstance(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc}, _shape{scene->load_shape(desc->property_node("shape"))} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return false; }
    [[nodiscard]] bool has_normal() const noexcept override { return false; }
    [[nodiscard]] bool has_uv() const noexcept override { return false; }
    [[nodiscard]] span<const Vertex> vertices() const noexcept override { return {}; }
    [[nodiscard]] span<const Triangle> triangles() const noexcept override { return {}; }
    [[nodiscard]] span<const Shape *const> children() const noexcept override { return {&_shape, 1u}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ShapeInstance)
