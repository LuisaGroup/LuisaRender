//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "node.h"
#include "acceleration.h"
#include "transform.h"

namespace luisa {
class Shape;
class Geometry;
}

namespace luisa {

class GeometryView {

private:
    Geometry *_geometry;
    
    
};

class Geometry {

public:
    friend class Shape;

private:
    std::vector<float3> _positions;
    std::vector<float3> _normals;
    std::vector<float2> _tex_coords;
    std::vector<uint32_t> _position_indices;
    std::vector<uint32_t> _normal_indices;
    std::vector<uint32_t> _tex_coord_indices;
    std::unique_ptr<Acceleration> _acceleration;

public:

    
};

}
