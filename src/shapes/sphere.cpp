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
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;

public:
    SphereGeometry() noexcept = default;
    SphereGeometry(luisa::vector<Vertex> vertices,
                   luisa::vector<Triangle> triangles) noexcept
        : _vertices{std::move(vertices)},
          _triangles{std::move(triangles)} {}

public:
    [[nodiscard]] auto mesh() const noexcept { return MeshView{_vertices, _triangles}; }
    [[nodiscard]] static auto create(uint subdiv) noexcept {
        static constexpr auto direction_to_uv = [](float3 w) noexcept {
            auto theta = acos(w.y);
            auto phi = atan2(w.x, w.z);
            return fract(make_float2(.5f * inv_pi * phi, theta * inv_pi));
        };
        static constexpr auto spherical_tangent = [](float3 w) noexcept {
            if (w.y > 1.f - 1e-8f) { return make_float3(1.f, 0.f, 0.f); }
            return normalize(make_float3(-w.z, 0.f, w.x));
        };
        static auto base_vertices = [] {
            std::array<Vertex, sphere_base_vertices.size()> bv{};
            for (auto i = 0u; i < sphere_base_vertices.size(); i++) {
                auto p = normalize(sphere_base_vertices[i]);
                bv[i] = Vertex::encode(p, p, make_float2());
            }
            return bv;
        }();
        LUISA_ASSERT(subdiv <= sphere_max_subdivision_level, "Subdivision level {} is too high.", subdiv);
        static std::array<std::shared_future<SphereGeometry>, sphere_max_subdivision_level + 1u> cache;
        static std::mutex mutex;
        std::scoped_lock lock{mutex};
        if (auto g = cache.at(subdiv); g.valid()) { return g; }
        auto future = ThreadPool::global().async([subdiv] {
            auto [vertices, triangles, _] = loop_subdivide(base_vertices, sphere_base_triangles, subdiv);
            for (auto &v : vertices) {
                auto p = normalize(v.position());
                auto uv = direction_to_uv(v.position());
                v = Vertex::encode(p, p, uv);
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
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return _geometry.get().mesh(); }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<Sphere>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereWrapper)
