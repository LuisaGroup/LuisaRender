//
// Created by Mike Smith on 2022/1/9.
//

#include <luisa-compute.h>
#include <base/shape.h>
#include <base/scene.h>

namespace luisa::render {

class ShapeGroup final : public Shape {

private:
    luisa::vector<const Shape *> _children;

public:
    ShapeGroup(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc} {
        auto shapes = desc->property_node_list("shapes");
        _children.reserve(shapes.size());
        for (auto shape : shapes) {
            _children.emplace_back(scene->load_shape(shape));
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return "group"; }
    [[nodiscard]] bool is_mesh() const noexcept override { return false; }
    [[nodiscard]] bool is_virtual() const noexcept override { return false; }
    [[nodiscard]] span<const float3> positions() const noexcept override { return {}; }
    [[nodiscard]] span<const VertexAttribute> attributes() const noexcept override { return {}; }
    [[nodiscard]] span<const Triangle> triangles() const noexcept override { return {}; }
    [[nodiscard]] span<const Shape *const> children() const noexcept override { return _children; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ShapeGroup)
