//
// Created by Mike Smith on 2022/11/8.
//

#include <base/shape.h>
#include <util/loop_subdiv.h>

namespace luisa::render {

static constexpr auto sphere_max_subdivision_level = 8u;

// Icosahedron
static constexpr std::array sphere_base_vertices{
    make_float3(0.f, -0.525731f, 0.850651f),
    make_float3(0.850651f, 0.f, 0.525731f),
    make_float3(0.850651f, 0.f, -0.525731f),
    make_float3(-0.850651f, 0.f, -0.525731f),
    make_float3(-0.850651f, 0.f, 0.525731f),
    make_float3(-0.525731f, 0.850651f, 0.f),
    make_float3(0.525731f, 0.850651f, 0.f),
    make_float3(0.525731f, -0.850651f, 0.f),
    make_float3(-0.525731f, -0.850651f, 0.f),
    make_float3(0.f, -0.525731f, -0.850651f),
    make_float3(0.f, 0.525731f, -0.850651f),
    make_float3(0.f, 0.525731f, 0.850651f)};

static constexpr std::array sphere_base_triangles{
    Triangle{1u, 2u, 6u},
    Triangle{1u, 7u, 2u},
    Triangle{3u, 4u, 5u},
    Triangle{4u, 3u, 8u},
    Triangle{6u, 5u, 11u},
    Triangle{5u, 6u, 10u},
    Triangle{9u, 10u, 2u},
    Triangle{10u, 9u, 3u},
    Triangle{7u, 8u, 9u},
    Triangle{8u, 7u, 0u},
    Triangle{11u, 0u, 1u},
    Triangle{0u, 11u, 4u},
    Triangle{6u, 2u, 10u},
    Triangle{1u, 6u, 11u},
    Triangle{3u, 5u, 10u},
    Triangle{5u, 4u, 11u},
    Triangle{2u, 7u, 9u},
    Triangle{7u, 1u, 0u},
    Triangle{3u, 9u, 8u},
    Triangle{4u, 8u, 0u}};

class SphereGeometry {

private:
    luisa::vector<Shape::Vertex> _vertices;
    luisa::vector<Triangle> _triangles;

private:
    SphereGeometry(luisa::vector<Shape::Vertex> vertices,
                   luisa::vector<Triangle> triangles) noexcept
        : _vertices{std::move(vertices)}, _triangles{std::move(triangles)} {}

public:
    [[nodiscard]] auto vertices() const noexcept { return luisa::span{_vertices}; }
    [[nodiscard]] auto triangles() const noexcept { return luisa::span{_triangles}; }
    [[nodiscard]] static auto create(uint subdiv) noexcept {
        LUISA_ASSERT(subdiv <= sphere_max_subdivision_level, "Subdivision level {} is too high.", subdiv);
        static std::array<std::shared_future<SphereGeometry>, sphere_max_subdivision_level + 1u> cache;
        static std::mutex mutex;
        std::scoped_lock lock{mutex};
        if (auto g = cache.at(subdiv); g.valid()) { return g; }
        auto future = ThreadPool::global().async([subdiv] {
            auto [positions, triangles] = loop_subdivide(
                sphere_base_vertices, sphere_base_triangles, subdiv);
            luisa::vector<Shape::Vertex> vertices(positions.size());
            for (auto i = 0u; i < positions.size(); i++) {
                auto direction_to_uv = [](float3 w) noexcept {
                    auto theta = acos(w.y);
                    auto phi = atan2(w.x, w.z);
                    return fract(make_float2(.5f * inv_pi * phi, theta * inv_pi));
                };
                auto w = normalize(positions[i]);
                vertices[i] = Shape::Vertex::encode(w, w, direction_to_uv(w));
            }
            return SphereGeometry{std::move(vertices), std::move(triangles)};
        });
        cache[subdiv] = future;
        return future;
    }
};

class Sphere : public Shape {

private:
    std::shared_future<SphereGeometry> _geometry;

public:
    Sphere(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _geometry{SphereGeometry::create(
              std::min(desc->property_uint_or_default("subdivision", 0u),
                       sphere_max_subdivision_level))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const Vertex> vertices() const noexcept override { return _geometry.get().vertices(); }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return _geometry.get().triangles(); }
    [[nodiscard]] bool has_normal() const noexcept override { return true; }
    [[nodiscard]] bool has_uv() const noexcept override { return true; }
};

using SphereWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<Sphere>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereWrapper)
