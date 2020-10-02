//
// Created by Mike Smith on 2020/10/2.
//

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/Subdivision.h>
#include <assimp/scene.h>

#include <render/shape.h>

namespace luisa::render::shape {

class TriangleMesh : public Shape {

private:
    static void _load(const std::filesystem::path &path, std::vector<Vertex> &vertices, std::vector<TriangleHandle> &indices, uint subdiv_level) noexcept {
        
        Assimp::Importer ai_importer;
        ai_importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                       aiComponent_COLORS |
                                       aiComponent_BONEWEIGHTS |
                                       aiComponent_ANIMATIONS |
                                       aiComponent_LIGHTS |
                                       aiComponent_CAMERAS |
                                       aiComponent_TEXTURES |
                                       aiComponent_MATERIALS);
        
        LUISA_INFO("Loading triangle mesh: ", path);
        auto ai_scene = ai_importer.ReadFile(path.string().c_str(),
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_GenNormals |
                                             aiProcess_PreTransformVertices |
                                             aiProcess_ImproveCacheLocality |
                                             aiProcess_FixInfacingNormals |
                                             aiProcess_FindInvalidData |
                                             aiProcess_GenUVCoords |
                                             aiProcess_TransformUVCoords |
                                             aiProcess_OptimizeMeshes |
                                             aiProcess_OptimizeGraph |
                                             aiProcess_FlipUVs);
        
        LUISA_EXCEPTION_IF(ai_scene == nullptr || (ai_scene->mFlags & static_cast<uint>(AI_SCENE_FLAGS_INCOMPLETE)) || ai_scene->mRootNode == nullptr,
                           "Failed to load triangle mesh: ", ai_importer.GetErrorString());
        
        std::vector<aiMesh *> ai_meshes(ai_scene->mNumMeshes);
        if (subdiv_level != 0u) {
            auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
            subdiv->Subdivide(ai_scene->mMeshes, ai_scene->mNumMeshes, ai_meshes.data(), subdiv_level);
        } else {
            std::copy(ai_scene->mMeshes, ai_scene->mMeshes + ai_scene->mNumMeshes, ai_meshes.begin());
        }
        
        for (auto ai_mesh : ai_meshes) {
            auto vertex_offset = static_cast<uint>(vertices.size());
            for (auto i = 0u; i < ai_mesh->mNumVertices; i++) {
                auto ai_position = ai_mesh->mVertices[i];
                auto ai_normal = ai_mesh->mNormals[i];
                Vertex vertex;
                vertex.position = make_float3(ai_position.x, ai_position.y, ai_position.z);
                vertex.normal = make_float3(ai_normal.x, ai_normal.y, ai_normal.z);
                if (ai_mesh->mTextureCoords[0] != nullptr) {
                    auto ai_tex_coord = ai_mesh->mTextureCoords[0][i];
                    vertex.uv = make_float2(ai_tex_coord.x, ai_tex_coord.y);
                }
                vertices.emplace_back(vertex);
            }
            for (auto f = 0u; f < ai_mesh->mNumFaces; f++) {
                auto ai_face = ai_mesh->mFaces[f];
                if (ai_face.mNumIndices == 3) {
                    indices.emplace_back(TriangleHandle{
                        vertex_offset + ai_face.mIndices[0],
                        vertex_offset + ai_face.mIndices[1],
                        vertex_offset + ai_face.mIndices[2]});
                } else if (ai_face.mNumIndices == 4) {
                    indices.emplace_back(TriangleHandle{
                        vertex_offset + ai_face.mIndices[0],
                        vertex_offset + ai_face.mIndices[1],
                        vertex_offset + ai_face.mIndices[2]});
                    indices.emplace_back(TriangleHandle{
                        vertex_offset + ai_face.mIndices[0],
                        vertex_offset + ai_face.mIndices[2],
                        vertex_offset + ai_face.mIndices[3]});
                } else {
                    LUISA_EXCEPTION("Only triangles and quads supported: ", ai_mesh->mName.data);
                }
            }
        }
    }

public:
    TriangleMesh(Device *device, const ParameterSet &params) : Shape{device, params} {
        auto subdiv = params["subdiv"].parse_uint_or_default(0u);
        auto path = std::filesystem::canonical(device->context().input_path(params["path"].parse_string()));
        _load(path, _vertices, _triangles, subdiv);
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::shape::TriangleMesh)
