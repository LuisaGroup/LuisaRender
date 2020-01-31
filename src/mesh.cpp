//
// Created by Mike Smith on 2019/10/19.
//

#include <iostream>
#include <tinyobjloader/tiny_obj_loader.h>

#include <core/mathematics.h>

#include "mesh.h"

namespace luisa {

Mesh Mesh::load(const std::vector<MeshDescriptor> &mesh_list) {
    
    using namespace math;
    
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = true;
    
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<uint> material_ids;
    std::vector<MaterialData> materials;
    
    for (auto &&desc : mesh_list) {
        
        auto model_matrix = desc.transform;
        auto normal_matrix = math::transpose(math::inverse(make_float3x3(model_matrix)));
        
        tinyobj::ObjReader reader;
        reader.ParseFromFile(desc.path, config);
        
        auto attributes = reader.GetAttrib();
        
        if (!reader.Valid()) {
            std::cerr << "Failed to load: " << desc.path << std::endl;
        }
        for (auto &&shape : reader.GetShapes()) {
            std::cout << "Processing shape: " << shape.name << std::endl;
            for (auto &&index : shape.mesh.indices) {
                auto pi = index.vertex_index * 3u;
                auto p = glm::vec3{model_matrix * glm::vec4{attributes.vertices[pi], attributes.vertices[pi + 1], attributes.vertices[pi + 2], 1.0f}};
                positions.emplace_back(p);
                auto ni = index.normal_index * 3u;
                auto n = normal_matrix * glm::vec3{attributes.normals[ni], attributes.normals[ni + 1], attributes.normals[ni + 2]};
                normals.emplace_back(n);
            }
            auto material_id = static_cast<uint>(materials.size());
            for (auto i = 0ul; i < shape.mesh.material_ids.size(); i++) {
                material_ids.emplace_back(material_id);
            }
            materials.emplace_back(MaterialData{desc.albedo, desc.is_mirror});
        }
    }
    Mesh mesh;
    mesh.positions = std::move(positions);
    mesh.normals = std::move(normals);
    mesh.material_ids = std::move(material_ids);
    mesh.materials = std::move(materials);
    
    std::cout << "Total: " << mesh.material_ids.size() << " triangles" << std::endl;
    
    return mesh;
}

}
