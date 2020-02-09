//
// Created by Mike Smith on 2020/2/9.
//

#pragma once

#include "node.h"
#include "parser.h"

namespace luisa {

class Transform : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Transform);

public:
    Transform(Device *device, const ParameterSet &) : Node{device} {}
    [[nodiscard]] virtual float4x4 matrix(float time) const = 0;

};

}
