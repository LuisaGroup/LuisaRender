//
// Created by Mike on 2022/2/18.
//

#include <base/shape.h>

namespace luisa::render {

class InlineMesh : public Shape {

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;
    luisa::vector<float2> _uvs;
    uint _properties{};

public:
    InlineMesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc} {

        auto triangles = desc->property_uint_list("indices");
        auto positions = desc->property_float_list("positions");
        auto normals = desc->property_float_list_or_default("normals");
        auto uvs = desc->property_float_list_or_default("uvs");

        if (triangles.size() % 3u != 0u ||
            positions.size() % 3u != 0u ||
            normals.size() % 3u != 0u ||
            uvs.size() % 2u != 0u ||
            (!normals.empty() && normals.size() != positions.size()) ||
            (!uvs.empty() && uvs.size() / 2u != positions.size() / 3u)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid vertex or triangle count.");
        }
        _properties = (!uvs.empty() ? Shape::property_flag_has_vertex_uv : 0u) |
                      (!normals.empty() ? Shape::property_flag_has_vertex_normal : 0u);

        auto triangle_count = triangles.size() / 3u;
        auto vertex_count = positions.size() / 3u;
        _triangles.resize(triangle_count);
        for (auto i = 0u; i < triangle_count; i++) {
            auto t0 = triangles[i * 3u + 0u];
            auto t1 = triangles[i * 3u + 1u];
            auto t2 = triangles[i * 3u + 2u];
            assert(t0 < vertex_count && t1 < vertex_count && t2 < vertex_count);
            _triangles[i] = Triangle{t0, t1, t2};
        }
        _vertices.resize(vertex_count);
        for (auto i = 0u; i < vertex_count; i++) {
            auto p0 = positions[i * 3u + 0u];
            auto p1 = positions[i * 3u + 1u];
            auto p2 = positions[i * 3u + 2u];
            auto p = make_float3(p0, p1, p2);
            auto n = normals.empty() ?
                         make_float3(0.f, 0.f, 1.f) :
                         make_float3(normals[i * 3u + 0u], normals[i * 3u + 1u], normals[i * 3u + 2u]);
            auto uv = uvs.empty() ? make_float2(0.f) : make_float2(uvs[i * 2u + 0u], uvs[i * 2u + 1u]);
            _vertices[i] = Vertex::encode(p, n, uv);
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return {_vertices, _triangles}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] uint vertex_properties() const noexcept override { return _properties; }
};

using InlineMeshWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<InlineMesh>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::InlineMeshWrapper)
