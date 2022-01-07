//
// Created by Mike on 2022/1/7.
//

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <core/thread_pool.h>
#include <scene/shape.h>

namespace luisa::render {

class TriangleMeshLoader {

private:
    luisa::vector<float3> _positions;
    luisa::vector<VertexAttribute> _attributes;
    luisa::vector<Triangle> _triangles;

public:
    [[nodiscard]] auto positions() const noexcept { return luisa::span{_positions}; }
    [[nodiscard]] auto attributes() const noexcept { return luisa::span{_attributes}; }
    [[nodiscard]] auto triangles() const noexcept { return luisa::span{_triangles}; }

    [[nodiscard]] static auto load(std::filesystem::path path) noexcept {
        return ThreadPool::global().async([path = std::move(path)] {
            TriangleMeshLoader mesh;
            LUISA_INFO("Loading mesh from '{}'.", path.string());
            return mesh;
        });
    }
};

class TriangleMesh final : public Shape {

private:
    std::shared_future<TriangleMeshLoader> _loader;

public:
    TriangleMesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc}, _loader{TriangleMeshLoader::load(desc->property_path("file"))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "trianglemesh"; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool is_rigid() const noexcept override { return true; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const float3> positions() const noexcept override { return _loader.get().positions(); }
    [[nodiscard]] luisa::span<const VertexAttribute> attributes() const noexcept override { return _loader.get().attributes(); }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return _loader.get().triangles(); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::TriangleMesh)
