//
// Created by Mike on 2022/1/7.
//

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/Subdivision.h>

#include <core/thread_pool.h>
#include <base/shape.h>

namespace luisa::render {

class MeshLoader {

private:
    luisa::vector<Shape::Vertex> _vertices;
    luisa::vector<Triangle> _triangles;
    bool _has_uv{};
    bool _has_normal{};

public:
    [[nodiscard]] auto vertices() const noexcept { return luisa::span{_vertices}; }
    [[nodiscard]] auto triangles() const noexcept { return luisa::span{_triangles}; }
    [[nodiscard]] auto has_normal() const noexcept { return _has_normal; }
    [[nodiscard]] auto has_uv() const noexcept { return _has_uv; }

    // Load the mesh from a file.
    [[nodiscard]] static auto load(std::filesystem::path path, uint subdiv_level,
                                   bool flip_uv, bool drop_normal) noexcept {

        // TODO: static lifetime seems not good...
        static luisa::lru_cache<uint64_t, std::shared_future<MeshLoader>> loaded_meshes{32u};
        static std::mutex mutex;

        auto abs_path = std::filesystem::canonical(path).string();
        auto key = luisa::hash64(abs_path, luisa::hash64(subdiv_level));

        std::scoped_lock lock{mutex};
        if (auto m = loaded_meshes.at(key)) {
            return *m;
        }

        auto future = ThreadPool::global().async([path = std::move(path), subdiv_level, flip_uv, drop_normal] {
            Clock clock;
            auto path_string = path.string();
            Assimp::Importer importer;
            importer.SetPropertyInteger(
                AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
            importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 45.f);
            auto import_flags = aiProcess_JoinIdenticalVertices | aiProcess_RemoveComponent |
                                aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes |
                                aiProcess_GenUVCoords | aiProcess_TransformUVCoords |
                                aiProcess_SortByPType | aiProcess_ValidateDataStructure |
                                aiProcess_ImproveCacheLocality | aiProcess_PreTransformVertices;
            if (!flip_uv) { import_flags |= aiProcess_FlipUVs; }
            import_flags |= drop_normal ? aiProcess_DropNormals : aiProcess_GenSmoothNormals;
            auto remove_flags = aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                                aiComponent_CAMERAS | aiComponent_COLORS |
                                aiComponent_LIGHTS | aiComponent_MATERIALS |
                                aiComponent_TEXTURES | aiComponent_TANGENTS_AND_BITANGENTS;
            importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(remove_flags));
            if (subdiv_level == 0) { import_flags |= aiProcess_Triangulate; }
            auto model = importer.ReadFile(path_string.c_str(), import_flags);
            if (model == nullptr || (model->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
                model->mRootNode == nullptr || model->mRootNode->mNumMeshes == 0) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to load mesh '{}': {}.",
                    path_string, importer.GetErrorString());
            }
            if (auto err = importer.GetErrorString();
                err != nullptr && err[0] != '\0') [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Mesh '{}' has warnings: {}.",
                    path_string, err);
            }
            LUISA_ASSERT(model->mNumMeshes == 1u, "Only single mesh is supported.");
            auto mesh = model->mMeshes[0];
            if (subdiv_level > 0u) {
                auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
                aiMesh *subdiv_mesh = nullptr;
                subdiv->Subdivide(mesh, subdiv_mesh, subdiv_level, true);
                model->mMeshes[0] = nullptr;
                mesh = subdiv_mesh;
                delete subdiv;
            }
            if (mesh->mTextureCoords[0] == nullptr ||
                mesh->mNumUVComponents[0] != 2) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid texture coordinates in mesh '{}': "
                    "address = {}, components = {}.",
                    path_string,
                    static_cast<void *>(mesh->mTextureCoords[0]),
                    mesh->mNumUVComponents[0]);
            }
            MeshLoader loader;
            auto vertex_count = mesh->mNumVertices;
            loader._vertices.resize(vertex_count);
            auto ai_positions = mesh->mVertices;
            auto ai_normals = mesh->mNormals;
            auto ai_tex_coords = mesh->mTextureCoords[0];
            loader._has_uv = ai_tex_coords != nullptr;
            loader._has_normal = !drop_normal && ai_normals != nullptr;
            if (!loader._has_normal) {
                LUISA_WARNING_WITH_LOCATION(
                    "Mesh '{}' has no normal data, "
                    "which may cause incorrect shading.",
                    path_string);
            }
            for (auto i = 0; i < vertex_count; i++) {
                auto p = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
                auto n = loader._has_normal ?
                             normalize(make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z)) :
                             make_float3(0.f);
                auto uv = loader._has_uv ?
                              make_float2(ai_tex_coords[i].x, ai_tex_coords[i].y) :
                              make_float2(0.f);
                loader._vertices[i] = Shape::Vertex::encode(p, n, uv);
            }
            if (subdiv_level == 0u) {
                auto ai_triangles = mesh->mFaces;
                loader._triangles.resize(mesh->mNumFaces);
                std::transform(
                    ai_triangles, ai_triangles + mesh->mNumFaces, loader._triangles.begin(),
                    [](const aiFace &face) noexcept {
                        assert(face.mNumIndices == 3u);
                        return Triangle{face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                    });
            } else {
                auto ai_quads = mesh->mFaces;
                loader._triangles.resize(mesh->mNumFaces * 2u);
                for (auto i = 0u; i < mesh->mNumFaces; i++) {
                    auto &face = ai_quads[i];
                    assert(face.mNumIndices == 4u);
                    loader._triangles[i * 2u + 0u] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                    loader._triangles[i * 2u + 1u] = {face.mIndices[0], face.mIndices[2], face.mIndices[3]};
                }
            }
            LUISA_INFO(
                "Loaded triangle mesh '{}' in {} ms.",
                path_string, clock.toc());
            return loader;
        });
        loaded_meshes.emplace(key, future);
        return future;
    }
};

class Mesh final : public Shape {

private:
    std::shared_future<MeshLoader> _loader;

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _loader{MeshLoader::load(
              desc->property_path("file"),
              desc->property_uint_or_default("subdivision", 0u),
              desc->property_bool_or_default("flip_uv", false),
              desc->property_bool_or_default("drop_normal", false))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const Vertex> vertices() const noexcept override { return _loader.get().vertices(); }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return _loader.get().triangles(); }
    [[nodiscard]] bool has_normal() const noexcept override { return _loader.get().has_normal(); }
    [[nodiscard]] bool has_uv() const noexcept override { return _loader.get().has_uv(); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Mesh)
