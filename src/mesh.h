//
// Created by Mike Smith on 2019/10/19.
//

#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <tinyobjloader/tiny_obj_loader.h>

class Mesh {

private:
    std::vector<glm::vec4> _positions;  // for padding
    std::vector<glm::vec4> _normals;
    std::vector<glm::vec2> _tex_coords;
    std::vector<uint32_t> _material_ids;

};


