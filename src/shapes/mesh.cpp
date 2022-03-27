//
// Created by Mike on 2022/1/7.
//

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>

#include <core/thread_pool.h>
#include <base/shape.h>

namespace luisa::render {

class MeshLoader {

private:
    luisa::vector<Shape::Vertex> _vertices;
    luisa::vector<Triangle> _triangles;
    bool _has_uv{};

public:
    [[nodiscard]] auto vertices() const noexcept { return luisa::span{_vertices}; }
    [[nodiscard]] auto triangles() const noexcept { return luisa::span{_triangles}; }
    [[nodiscard]] auto has_uv() const noexcept { return _has_uv; }

    [[nodiscard]] static auto load(std::filesystem::path path) noexcept {

        return ThreadPool::global().async([path = std::move(path)] {
            Clock clock;
            auto path_string = path.string();
            Assimp::Importer importer;
            importer.SetPropertyInteger(
                AI_CONFIG_PP_RVC_FLAGS,
                aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                    aiComponent_CAMERAS | aiComponent_COLORS |
                    aiComponent_LIGHTS | aiComponent_MATERIALS |
                    aiComponent_TEXTURES);
            importer.SetPropertyInteger(
                AI_CONFIG_PP_SBP_REMOVE,
                aiPrimitiveType_LINE | aiPrimitiveType_POINT);
            importer.SetPropertyBool(
                AI_CONFIG_PP_FD_CHECKAREA, false);
            auto model = importer.ReadFile(
                path_string.c_str(),
                aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
                    aiProcess_RemoveComponent | aiProcess_ImproveCacheLocality |
                    aiProcess_OptimizeGraph | aiProcess_GenNormals |
                    aiProcess_GenUVCoords | aiProcess_CalcTangentSpace |
                    aiProcess_FixInfacingNormals | aiProcess_RemoveRedundantMaterials |
                    aiProcess_FindInvalidData | aiProcess_FlipUVs | aiProcess_TransformUVCoords |
                    aiProcess_SortByPType | aiProcess_FindDegenerates);
            if (model == nullptr || (model->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
                model->mRootNode == nullptr || model->mRootNode->mNumMeshes == 0) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to load mesh '{}': {}.",
                    path_string, importer.GetErrorString());
            }
            auto mesh = model->mMeshes[0];
            if (auto uv_count = std::count_if(
                    std::cbegin(mesh->mTextureCoords),
                    std::cend(mesh->mTextureCoords),
                    [](auto p) noexcept { return p != nullptr; });
                uv_count > 1) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "More than one set of texture coordinates "
                    "found in mesh '{}'. Only the first set "
                    "will be considered.",
                    path_string);
            }
            if (mesh->mTextureCoords[0] == nullptr ||
                mesh->mNumUVComponents[0] != 2) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid texture coordinates in mesh '{}': "
                    "address = {}, components = {}.",
                    path_string,
                    fmt::ptr(mesh->mTextureCoords[0]),
                    mesh->mNumUVComponents[0]);
            }
            MeshLoader loader;
            auto vertex_count = mesh->mNumVertices;
            auto ai_positions = mesh->mVertices;
            auto ai_normals = mesh->mNormals;
            auto ai_tex_coords = mesh->mTextureCoords[0];
            auto ai_bitangents = mesh->mBitangents;
            if (ai_bitangents == nullptr) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid tangent space in mesh '{}'.",
                    path_string);
            }
            loader._vertices.resize(vertex_count);
            auto compute_tangent = [ai_bitangents](auto i, float3 n) noexcept {
                if (ai_bitangents == nullptr) {
                    auto b = abs(n.x) > abs(n.z) ?
                                 make_float3(-n.y, n.x, 0.0f) :
                                 make_float3(0.0f, -n.z, n.y);
                    return normalize(cross(b, n));
                }
                auto bitangent = make_float3(
                    ai_bitangents[i].x, ai_bitangents[i].y, ai_bitangents[i].z);
                return normalize(cross(bitangent, n));
            };
            loader._has_uv = ai_tex_coords != nullptr;
            auto compute_uv = [ai_tex_coords](auto i) noexcept {
                if (ai_tex_coords == nullptr) { return make_float2(); }
                return make_float2(ai_tex_coords[i].x, ai_tex_coords[i].y);
            };
            for (auto i = 0; i < vertex_count; i++) {
                auto p = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
                auto n = make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z);
                auto t = compute_tangent(i, n);
                auto uv = compute_uv(i);
                loader._vertices[i] = Shape::Vertex::encode(p, n, t, uv);
            }
            auto triangle_count = mesh->mNumFaces;
            auto ai_triangles = mesh->mFaces;
            loader._triangles.resize(triangle_count);
            std::transform(
                ai_triangles, ai_triangles + triangle_count,
                loader._triangles.begin(),
                [](const aiFace &face) noexcept {
                    auto t = face.mIndices;
                    return Triangle{t[0], t[1], t[2]};
                });
            LUISA_INFO(
                "Loaded triangle mesh '{}' in {} ms.",
                path_string, clock.toc());
            return loader;
        });
    }
};

class Mesh final : public Shape {

private:
    std::shared_future<MeshLoader> _loader;
    AccelBuildHint _build_hint{AccelBuildHint::FAST_TRACE};

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc}, _loader{MeshLoader::load(desc->property_path("file"))} {
        auto hint = desc->property_string_or_default("build_hint", "");
        if (hint == "fast_update") {
            _build_hint = AccelBuildHint::FAST_UPDATE;
        } else if (hint == "fast_rebuild") {
            _build_hint = AccelBuildHint::FAST_REBUILD;
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const Vertex> vertices() const noexcept override { return _loader.get().vertices(); }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return _loader.get().triangles(); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Mesh)
