//
// Created by Mike Smith on 2020/2/4.
//

#pragma once

#include "node.h"
#include "parser.h"
#include "geometry.h"
#include "illumination.h"

namespace luisa {

class Render : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Render);

protected:
    float2 _time_span;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<Integrator> _integrator;
    std::unique_ptr<Geometry> _geometry;
    std::unique_ptr<Illumination> _illumination;

public:
    Render(Device *device, const ParameterSet &parameter_set[[maybe_unused]])
        : Node{device},
          _time_span{parameter_set["time_span"].parse_float2_or_default(make_float2(0.0f))},
          _camera{parameter_set["camera"].parse<Camera>()},
          _integrator{parameter_set["integrator"].parse<Integrator>()} {
        
        _geometry = Geometry::create(_device, parameter_set["shapes"].parse_reference_list<Shape>(), _time_span.x);
        _illumination = Illumination::create(_device, parameter_set["lights"].parse_reference_list<Light>());
    }
    
    virtual void execute() = 0;
    
};

}
