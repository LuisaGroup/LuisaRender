//
// Created by Mike Smith on 2022/1/9.
//

#include <luisa-compute.h>
#include <base/shape.h>
#include <base/scene.h>

namespace luisa::render {

class ShapeGroup : public Shape {

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
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] span<const Shape *const> children() const noexcept override { return _children; }
};

using GroupWrapper = VisibilityShapeWrapper<ShapeGroup>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GroupWrapper)
