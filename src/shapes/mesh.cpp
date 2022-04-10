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

public:
    [[nodiscard]] auto vertices() const noexcept { return luisa::span{_vertices}; }
    [[nodiscard]] auto triangles() const noexcept { return luisa::span{_triangles}; }
    [[nodiscard]] auto has_uv() const noexcept { return _has_uv; }

    [[nodiscard]] static auto load(std::filesystem::path path, uint subdiv_level) noexcept {

        return ThreadPool::global().async([path = std::move(path), subdiv_level] {
            Clock clock;
            auto path_string = path.string();
            Assimp::Importer importer;
            importer.SetPropertyInteger(
                AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
            importer.SetPropertyBool(AI_CONFIG_PP_FD_CHECKAREA, false);
            auto import_flags = aiProcess_JoinIdenticalVertices | aiProcess_RemoveComponent |
                                aiProcess_OptimizeGraph | aiProcess_GenUVCoords |
                                aiProcess_FlipUVs | aiProcess_TransformUVCoords |
                                aiProcess_FixInfacingNormals | aiProcess_RemoveRedundantMaterials |
                                aiProcess_FindInvalidData | aiProcess_SortByPType | aiProcess_GenNormals |
                                aiProcess_FindDegenerates | aiProcess_ImproveCacheLocality;
            auto remove_flags = aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                                aiComponent_CAMERAS | aiComponent_COLORS |
                                aiComponent_LIGHTS | aiComponent_MATERIALS |
                                aiComponent_TEXTURES;
            if (subdiv_level == 0) { import_flags |= aiProcess_Triangulate; }
            importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(remove_flags));
            auto model = importer.ReadFile(path_string.c_str(), import_flags);
            if (model == nullptr || (model->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
                model->mRootNode == nullptr || model->mRootNode->mNumMeshes == 0) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to load mesh '{}': {}.",
                    path_string, importer.GetErrorString());
            }
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
            luisa::vector<aiVector3t<ai_real>> computed_normals;
            auto vertex_count = mesh->mNumVertices;
            auto ai_positions = mesh->mVertices;
            auto ai_normals = mesh->mNormals;
            auto ai_tex_coords = mesh->mTextureCoords[0];
            constexpr auto to_float3 = [](aiVector3t<ai_real> v) noexcept {
                return make_float3(v.x, v.y, v.z);
            };
            loader._vertices.resize(vertex_count);
            loader._has_uv = ai_tex_coords != nullptr;
            auto compute_uv = [ai_tex_coords](auto i) noexcept {
                if (ai_tex_coords == nullptr) { return make_float2(); }
                return make_float2(ai_tex_coords[i].x, ai_tex_coords[i].y);
            };
            for (auto i = 0; i < vertex_count; i++) {
                auto p = to_float3(ai_positions[i]);
                auto n = to_float3(ai_normals[i]);
                loader._vertices[i].pos = p;
                loader._vertices[i].compressed_normal =
                    Shape::Vertex::oct_encode(normalize(n));
                auto uv = compute_uv(i);
                loader._vertices[i].compressed_uv[0] = uv.x;
                loader._vertices[i].compressed_uv[1] = uv.y;
            }
            if (subdiv_level == 0u) {
                auto ai_triangles = mesh->mFaces;
                loader._triangles.resize(mesh->mNumFaces);
                std::transform(
                    ai_triangles, ai_triangles + mesh->mNumFaces, loader._triangles.begin(),
                    [](const aiFace &face) noexcept {
                        return Triangle{face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                    });
            } else {
                auto ai_quads = mesh->mFaces;
                loader._triangles.resize(mesh->mNumFaces * 2u);
                for (auto i = 0u; i < mesh->mNumFaces; i++) {
                    auto &face = ai_quads[i];
                    loader._triangles[i * 2u + 0u] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                    loader._triangles[i * 2u + 1u] = {face.mIndices[0], face.mIndices[2], face.mIndices[3]};
                }
            }

            // compute tangent space (code adapted from PBRT-v4).
            if (loader._has_uv) {
                auto difference_of_products = [](auto a, auto b, auto c, auto d) noexcept {
                    auto cd = c * d;
                    auto differenceOfProducts = a * b - cd;
                    auto error = -c * d + cd;
                    return differenceOfProducts + error;
                };
                for (auto t : loader._triangles) {
                    auto &&v0 = loader._vertices[t.i0];
                    auto &&v1 = loader._vertices[t.i1];
                    auto &&v2 = loader._vertices[t.i2];
                    std::array uv{make_float2(v0.compressed_uv[0], v0.compressed_uv[1]),
                                  make_float2(v1.compressed_uv[0], v1.compressed_uv[1]),
                                  make_float2(v2.compressed_uv[0], v2.compressed_uv[1])};
                    auto duv02 = uv[0] - uv[2];
                    auto duv12 = uv[1] - uv[2];
                    auto dp02 = v0.pos - v2.pos;
                    auto dp12 = v1.pos - v2.pos;
                    auto det = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);
                    auto dpdu = make_float3();
                    auto dpdv = make_float3();
                    auto degenerate_uv = std::abs(det) < 1e-9f;
                    if (!degenerate_uv) [[likely]] {
                        // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
                        auto invdet = 1.f / det;
                        dpdu = difference_of_products(duv12[1], dp02, duv02[1], dp12) * invdet;
                        dpdv = difference_of_products(duv02[0], dp12, duv12[0], dp02) * invdet;
                    }
                    auto length_squared = [](auto v) noexcept { return dot(v, v); };
                    // Handle degenerate triangle $(u,v)$ parameterization or partial derivatives
                    if (degenerate_uv || length_squared(cross(dpdu, dpdv)) == 0.f) [[unlikely]] {
                        auto n = cross(v2.pos - v0.pos, v1.pos - v0.pos);
                        auto b = abs(n.x) > abs(n.z) ?
                                     make_float3(-n.y, n.x, 0.0f) :
                                     make_float3(0.0f, -n.z, n.y);
                        dpdu = cross(b, n);
                    }
                    loader._vertices[t.i0].compressed_tangent = Shape::Vertex::oct_encode(dpdu);
                    loader._vertices[t.i1].compressed_tangent = Shape::Vertex::oct_encode(dpdu);
                    loader._vertices[t.i2].compressed_tangent = Shape::Vertex::oct_encode(dpdu);
                }
            } else {
                for (auto i = 0u; i < vertex_count; i++) {
                    auto n = to_float3(ai_normals[i]);
                    auto b = abs(n.x) > abs(n.z) ?
                                 make_float3(-n.y, n.x, 0.0f) :
                                 make_float3(0.0f, -n.z, n.y);
                    loader._vertices[i].compressed_tangent =
                        Shape::Vertex::oct_encode(normalize(cross(b, n)));
                }
            }
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

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _loader{MeshLoader::load(
              desc->property_path("file"),
              desc->property_uint_or_default("subdivision", 0u))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const Vertex> vertices() const noexcept override { return _loader.get().vertices(); }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return _loader.get().triangles(); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Mesh)
