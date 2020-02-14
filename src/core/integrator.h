//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "node.h"
#include "parser.h"
#include "ray.h"

namespace luisa {

class Integrator : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Integrator);

protected:
    uint _spp;

public:
    Integrator(Device *device, const ParameterSet &parameter_set)
        : Node{device},
          _spp{parameter_set["spp"].parse_uint_or_default(1024u)} {}
    
    [[nodiscard]] uint spp() const noexcept { return _spp; }
    
    virtual void render() = 0;
    
};

}
