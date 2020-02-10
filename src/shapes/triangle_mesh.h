//
// Created by Mike Smith on 2020/2/10.
//

#pragma once

#include <filesystem>
#include <core/shape.h>

namespace luisa {

class TriangleMesh : public Shape {

private:
    std::filesystem::path _path;
    uint _subdiv_level;

public:
    void load(GeometryEncoder encoder) override;
    TriangleMesh(Device *device, const ParameterSet &parameter_set);

};

LUISA_REGISTER_NODE_CREATOR("TriangleMesh", TriangleMesh)

}
