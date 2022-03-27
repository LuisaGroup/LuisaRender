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
                    aiProcess_GenUVCoords | aiProcess_FlipUVs |
                    aiProcess_TransformUVCoords | aiProcess_FixInfacingNormals |
                    aiProcess_RemoveRedundantMaterials | aiProcess_FindInvalidData |
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
            loader._vertices.resize(vertex_count);
            loader._has_uv = ai_tex_coords != nullptr;
            auto compute_uv = [ai_tex_coords](auto i) noexcept {
                if (ai_tex_coords == nullptr) { return make_float2(); }
                return make_float2(ai_tex_coords[i].x, ai_tex_coords[i].y);
            };
            for (auto i = 0; i < vertex_count; i++) {
                auto p = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
                auto n = make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z);
                loader._vertices[i].pos = p;
                loader._vertices[i].compressed_normal = Shape::Vertex::oct_encode(n);
                auto uv = compute_uv(i);
                loader._vertices[i].compressed_uv[0] = uv.x;
                loader._vertices[i].compressed_uv[1] = uv.y;
            }
            auto triangle_count = mesh->mNumFaces;
            auto ai_triangles = mesh->mFaces;
            loader._triangles.resize(triangle_count);
            std::transform(
                ai_triangles, ai_triangles + triangle_count, loader._triangles.begin(),
                [](const aiFace &face) noexcept {
                    return Triangle{face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                });

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
                    auto n = make_float3(
                        ai_normals[i].x, ai_normals[i].y, ai_normals[i].z);
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
