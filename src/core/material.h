//
// Created by Mike Smith on 2020/2/4.
//

#pragma once

#include "node.h"

namespace luisa {

class Material : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Material);
    
};

}