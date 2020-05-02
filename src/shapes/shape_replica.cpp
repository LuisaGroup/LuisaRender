//
// Created by Mike Smith on 2020/2/10.
//

#include <core/shape.h>

namespace luisa {

class ShapeReplica : public Shape {

private:
    std::shared_ptr<Shape> _reference;

public:
    ShapeReplica(Device *device, const ParameterSet &parameter_set)
        : Shape{device, parameter_set} { _reference = parameter_set["reference"].parse<Shape>(); }
    
    void load(GeometryEncoder &encoder) override {
        encoder.replicate(this, _reference.get());
    }
    
};

LUISA_REGISTER_NODE_CREATOR("Replica", ShapeReplica)

}
