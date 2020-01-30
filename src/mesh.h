//
// Created by Mike Smith on 2019/10/19.
//

#pragma once

#include <vector>
#include <filesystem>
#include <core/data_types.h>

#include "material_data.h"

namespace luisa {

struct MeshDescriptor {
    std::filesystem::path path;
    glm::mat4 transform;
    glm::vec3 albedo;
    bool is_mirror;
};

struct Mesh {
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<uint> material_ids;
    std::vector<MaterialData> materials;
    static Mesh load(const std::vector<MeshDescriptor> &mesh_list);
};

}
