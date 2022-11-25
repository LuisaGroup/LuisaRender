//
// Created by Mike on 2022/1/7.
//

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/Subdivision.h>

#include <core/thread_pool.h>
#include <base/shape.h>

namespace luisa::render {

class MeshLoader {

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<float2> _uvs;
    luisa::vector<Triangle> _triangles;
    uint _properties{};

public:
    [[nodiscard]] auto mesh() const noexcept { return MeshView{_vertices, _uvs, _triangles}; }
    [[nodiscard]] auto properties() const noexcept { return _properties; }

    // Load the mesh from a file.
    [[nodiscard]] static auto load(std::filesystem::path path, uint subdiv_level,
                                   bool flip_uv, bool drop_normal, bool drop_uv) noexcept {

        // TODO: static lifetime seems not good...
        static luisa::lru_cache<uint64_t, std::shared_future<MeshLoader>> loaded_meshes{256u};
        static std::mutex mutex;

        auto abs_path = std::filesystem::canonical(path).string();
        auto key = luisa::hash64(abs_path, luisa::hash64(subdiv_level));

        std::scoped_lock lock{mutex};
        if (auto m = loaded_meshes.at(key)) { return *m; }

        auto future = ThreadPool::global().async([path = std::move(path), subdiv_level, flip_uv, drop_normal, drop_uv] {
            Clock clock;
            auto path_string = path.string();
            Assimp::Importer importer;
            importer.SetPropertyInteger(
                AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
            importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 45.f);
            auto import_flags = aiProcess_RemoveComponent | aiProcess_SortByPType |
                                aiProcess_ValidateDataStructure | aiProcess_ImproveCacheLocality |
                                aiProcess_PreTransformVertices | aiProcess_FindInvalidData |
                                aiProcess_JoinIdenticalVertices;
            auto remove_flags = aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                                aiComponent_CAMERAS | aiComponent_LIGHTS |
                                aiComponent_MATERIALS | aiComponent_TEXTURES |
                                aiComponent_COLORS | aiComponent_TANGENTS_AND_BITANGENTS;
            if (drop_uv) {
                remove_flags |= aiComponent_TEXCOORDS;
            } else {
                if (!flip_uv) { import_flags |= aiProcess_FlipUVs; }
                import_flags |= aiProcess_GenUVCoords | aiProcess_TransformUVCoords;
            }
            if (drop_normal) {
                import_flags |= aiProcess_DropNormals;
                remove_flags |= aiComponent_NORMALS;
            } else {
                import_flags |= aiProcess_GenSmoothNormals;
            }
            if (subdiv_level == 0) { import_flags |= aiProcess_Triangulate; }
            importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(remove_flags));
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
            auto vertex_count = mesh->mNumVertices;
            auto ai_positions = mesh->mVertices;
            auto ai_normals = mesh->mNormals;
            MeshLoader loader;
            loader._vertices.resize(vertex_count);
            if (ai_normals) { loader._properties |= Shape::property_flag_has_vertex_normal; }
            for (auto i = 0; i < vertex_count; i++) {
                auto p = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
                auto n = ai_normals ?
                             normalize(make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z)) :
                             make_float3(0.f, 0.f, 1.f);
                loader._vertices[i] = Vertex::encode(p, n);
            }
            if (auto ai_tex_coords = mesh->mTextureCoords[0]) {
                loader._uvs.resize(vertex_count);
                loader._properties |= Shape::property_flag_has_vertex_uv;
                for (auto i = 0; i < vertex_count; i++) {
                    loader._uvs[i] = make_float2(ai_tex_coords[i].x, ai_tex_coords[i].y);
                }
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
            LUISA_INFO("Loaded triangle mesh '{}' in {} ms.", path_string, clock.toc());
            return loader;
        });
        loaded_meshes.emplace(key, future);
        return future;
    }
};

class Mesh : public Shape {

private:
    std::shared_future<MeshLoader> _loader;

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _loader{MeshLoader::load(desc->property_path("file"),
                                   desc->property_uint_or_default("subdivision", 0u),
                                   desc->property_bool_or_default("flip_uv", false),
                                   desc->property_bool_or_default("drop_normal", false),
                                   desc->property_bool_or_default("drop_uv", false))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return _loader.get().mesh(); }
    [[nodiscard]] uint vertex_properties() const noexcept override { return _loader.get().properties(); }
};

using MeshWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<Mesh>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MeshWrapper)
