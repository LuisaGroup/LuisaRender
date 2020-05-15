//
// Created by Mike Smith on 2020/2/10.
//

#include <assimp/Importer.hpp>
#include <assimp/Subdivision.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <core/shape.h>

namespace luisa {

class TriangleMesh : public Shape {

private:
    std::string _path;
    uint _subdiv_level;

public:
    void load(GeometryEncoder &encoder) override;
    TriangleMesh(Device *device, const ParameterSet &parameter_set)
        : Shape{device, parameter_set},
          _path{parameter_set["path"].parse_string()},
          _subdiv_level{parameter_set["subdiv"].parse_uint_or_default(0u)} {}
};

void TriangleMesh::load(GeometryEncoder &encoder) {
    
    Assimp::Importer ai_importer;
    ai_importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                                   aiComponent_COLORS |
                                   aiComponent_BONEWEIGHTS |
                                   aiComponent_ANIMATIONS |
                                   aiComponent_LIGHTS |
                                   aiComponent_CAMERAS |
                                   aiComponent_TEXTURES |
                                   aiComponent_MATERIALS);
    
    auto path = _device->context().working_path(_path);
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
                                         aiProcess_FlipUVs);
    
    LUISA_EXCEPTION_IF(ai_scene == nullptr || (ai_scene->mFlags & static_cast<uint>(AI_SCENE_FLAGS_INCOMPLETE)) || ai_scene->mRootNode == nullptr,
                       "Failed to load triangle mesh: ", ai_importer.GetErrorString());
    
    std::vector<aiMesh *> ai_meshes(ai_scene->mNumMeshes);
    if (_subdiv_level != 0u) {
        auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
        subdiv->Subdivide(ai_scene->mMeshes, ai_scene->mNumMeshes, ai_meshes.data(), _subdiv_level);
    } else {
        std::copy(ai_scene->mMeshes, ai_scene->mMeshes + ai_scene->mNumMeshes, ai_meshes.begin());
    }
    
    auto m = _transform == nullptr ? math::identity() : _transform->static_matrix();
    auto n = transpose(inverse(make_float3x3(m)));
    
    for (auto ai_mesh : ai_meshes) {
        if (ai_mesh->mTextureCoords[0] == nullptr) {
            LUISA_WARNING("No texture coordinates in mesh, setting to (0, 0), name: \"", ai_mesh->mName.data, "\"");
            for (auto i = 0u; i < ai_mesh->mNumVertices; i++) {
                auto ai_position = ai_mesh->mVertices[i];
                auto ai_normal = ai_mesh->mNormals[i];
                auto position = make_float3(m * make_float4(ai_position.x, ai_position.y, ai_position.z, 1.0f));
                auto normal = normalize(n * make_float3(ai_normal.x, ai_normal.y, ai_normal.z));
                encoder.add_vertex(position, normal, make_float2());
            }
        } else {
            for (auto i = 0u; i < ai_mesh->mNumVertices; i++) {
                auto ai_position = ai_mesh->mVertices[i];
                auto ai_normal = ai_mesh->mNormals[i];
                auto ai_tex_coord = ai_mesh->mTextureCoords[0][i];
                auto position = make_float3(m * make_float4(ai_position.x, ai_position.y, ai_position.z, 1.0f));
                auto normal = normalize(n * make_float3(ai_normal.x, ai_normal.y, ai_normal.z));
                auto tex_coord = make_float2(ai_tex_coord.x, ai_tex_coord.y);
                encoder.add_vertex(position, normal, tex_coord);
            }
        }
        for (auto f = 0u; f < ai_mesh->mNumFaces; f++) {
            auto ai_face = ai_mesh->mFaces[f];
            if (ai_face.mNumIndices == 3) {
                encoder.add_indices(make_uint3(ai_face.mIndices[0], ai_face.mIndices[1], ai_face.mIndices[2]));
            } else if (ai_face.mNumIndices == 4) {
                encoder.add_indices(make_uint3(ai_face.mIndices[0], ai_face.mIndices[1], ai_face.mIndices[2]));
                encoder.add_indices(make_uint3(ai_face.mIndices[0], ai_face.mIndices[2], ai_face.mIndices[3]));
            } else {
                LUISA_EXCEPTION("Only triangles and quads supported: ", ai_mesh->mName.data);
            }
        }
    }
    encoder.create(this);
}

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::TriangleMesh)
