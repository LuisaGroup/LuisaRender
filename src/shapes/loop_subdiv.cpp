//
// Created by Mike Smith on 2022/11/8.
//

#include <base/shape.h>
#include <base/scene.h>
#include <util/loop_subdiv.h>

namespace luisa::render {

static constexpr auto max_loop_subdivision_level = 10u;

// TODO: preserve uv mapping
class LoopSubdiv : public Shape {

private:
    const Shape *_mesh;
    std::shared_future<std::pair<luisa::vector<Vertex>, luisa::vector<Triangle>>> _geometry;

public:
    LoopSubdiv(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _mesh{scene->load_shape(desc->property_node_or_default(
              "mesh", lazy_construct([desc] {
                  return desc->property_node_or_default(
                      "shape", lazy_construct([desc] { return desc->property_node("base"); }));
              })))} {
        LUISA_ASSERT(_mesh->is_mesh(), "LoopSubdiv only supports mesh shapes.");
        auto level = std::min(desc->property_uint_or_default("level", 1u),
                              max_loop_subdivision_level);
        if (level == 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "LoopSubdiv level is 0, which is equivalent to no subdivision.");
        } else {
            _geometry = ThreadPool::global().async([level, mesh = _mesh] {
                auto m = mesh->mesh();
                Clock clk;
                auto [vertices, triangles, _] = loop_subdivide(m.vertices, m.triangles, level);
                LUISA_INFO("LoopSubdiv (level = {}): subdivided {} vertices and {} "
                           "triangles into {} vertices and {} triangles in {} ms.",
                           level, m.vertices.size(), m.triangles.size(),
                           vertices.size(), triangles.size(), clk.toc());
                return std::make_pair(std::move(vertices), std::move(triangles));
            });
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override {
        return _geometry.valid() ?
                   MeshView{_geometry.get().first, {}, _geometry.get().second} :
                   _mesh->mesh();
    }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return _geometry.valid() ?
                   Shape::property_flag_has_vertex_normal :
                   _mesh->vertex_properties();
    }
    [[nodiscard]] AccelUsageHint build_hint() const noexcept override { return _mesh->build_hint(); }
};

using LoopSubdivWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<LoopSubdiv>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LoopSubdivWrapper)
