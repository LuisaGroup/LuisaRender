//
// Created by Mike Smith on 2022/11/8.
//

#include <base/shape.h>
#include <base/scene.h>
#include <util/loop_subdiv.h>

namespace luisa::render {

static constexpr auto max_loop_subdivision_level = 10u;

class LoopSubdiv : public Shape {

private:
    const Shape *_mesh;
    std::shared_future<std::pair<luisa::vector<Vertex>, luisa::vector<Triangle>>> _geometry;

public:
    LoopSubdiv(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _mesh{scene->load_shape(desc->property_node_or_default(
              "mesh", lazy_construct([desc] {
                  return desc->property_node("shape");
              })))} {
        LUISA_ASSERT(_mesh->is_mesh(), "LoopSubdiv only supports mesh shapes.");
        auto level = std::min(desc->property_uint_or_default("level", 1u),
                              max_loop_subdivision_level);
        if (level == 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "LoopSubdiv level is 0, which is equivalent to no subdivision.");
        } else {
            _geometry = ThreadPool::global().async([level, mesh = _mesh] {
                auto base_vertices = mesh->vertices();
                auto base_triangles = mesh->triangles();
                Clock clk;
                auto s = loop_subdivide(base_vertices, base_triangles, level);
                LUISA_INFO("LoopSubdiv (level = {}): subdivided {} vertices and {} "
                           "triangles into {} vertices and {} triangles in {} ms.",
                           level, base_vertices.size(), base_triangles.size(),
                           s.first.size(), s.second.size(), clk.toc());
                return s;
            });
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const Vertex> vertices() const noexcept override {
        return _geometry.valid() ? _geometry.get().first : _mesh->vertices();
    }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override {
        return _geometry.valid() ? _geometry.get().second : _mesh->triangles();
    }
    [[nodiscard]] bool has_normal() const noexcept override { return true; }
    [[nodiscard]] bool has_uv() const noexcept override { return _mesh->has_uv(); }
    [[nodiscard]] AccelUsageHint build_hint() const noexcept override { return _mesh->build_hint(); }
};

using LoopSubdivWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<LoopSubdiv>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LoopSubdivWrapper)
