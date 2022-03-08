//
// Created by Mike on 2022/2/18.
//

#include <base/shape.h>

namespace luisa::render {

class InlineMesh final : public Shape {

private:
    luisa::vector<float3> _positions;
    luisa::vector<VertexAttribute> _attributes;
    luisa::vector<Triangle> _triangles;

public:
    InlineMesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc} {

        auto triangles = desc->property_uint_list("indices");
        auto positions = desc->property_float_list("positions");
        auto normals = desc->property_float_list("normals");
        auto uvs = desc->property_float_list_or_default("uvs");
        auto tangents = desc->property_float_list_or_default("tangents");

        if (triangles.size() % 3u != 0u ||
            positions.size() % 3u != 0u ||
            normals.size() % 3u != 0u ||
            uvs.size() % 2u != 0u ||
            tangents.size() % 3u != 0u ||
            (positions.size() != normals.size()) ||
            (!uvs.empty() && uvs.size() / 2u != positions.size() / 3u) ||
            (!tangents.empty() && tangents.size() != positions.size())) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vertex or triangle count.");
        }

        auto triangle_count = triangles.size() / 3u;
        auto vertex_count = positions.size() / 3u;
        _positions.reserve(vertex_count);
        _attributes.reserve(vertex_count);
        _triangles.reserve(triangle_count);
        for (auto i = 0u; i < triangle_count; i++) {
            auto t0 = triangles[i * 3u + 0u];
            auto t1 = triangles[i * 3u + 1u];
            auto t2 = triangles[i * 3u + 2u];
            assert(t0 < vertex_count &&
                   t1 < vertex_count &&
                   t2 < vertex_count);
            _triangles.emplace_back(Triangle{t0, t1, t2});
        }
        for (auto i = 0u; i < vertex_count; i++) {
            auto p0 = positions[i * 3u + 0u];
            auto p1 = positions[i * 3u + 1u];
            auto p2 = positions[i * 3u + 2u];
            _positions.emplace_back(make_float3(p0, p1, p2));
            auto n0 = normals[i * 3u + 0u];
            auto n1 = normals[i * 3u + 1u];
            auto n2 = normals[i * 3u + 2u];
            auto normal = make_float3(n0, n1, n2);
            auto uv = make_float2();
            if (!uvs.empty()) {
                auto u = uvs[i * 2u + 0u];
                auto v = uvs[i * 2u + 1u];
                uv = make_float2(u, v);
            }
            auto tangent = make_float3();
            if (!tangents.empty()) {
                auto t0 = tangents[i * 3u + 0u];
                auto t1 = tangents[i * 3u + 1u];
                auto t2 = tangents[i * 3u + 2u];
                tangent = make_float3(t0, t1, t2);
            }
            _attributes.emplace_back(
                VertexAttribute::encode(
                    normal, tangent, uv));
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const float3> positions() const noexcept override { return _positions; }
    [[nodiscard]] luisa::span<const VertexAttribute> attributes() const noexcept override { return _attributes; }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return _triangles; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::InlineMesh)
