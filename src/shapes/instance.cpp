//
// Created by Mike Smith on 2020/2/10.
//

#include <core/shape.h>

namespace luisa {

class ShapeInstance : public Shape {

private:
    std::shared_ptr<Shape> _reference;

public:
    ShapeInstance(Device *device, const ParameterSet &parameter_set)
        : Shape{device, parameter_set} { _reference = parameter_set["reference"].parse<Shape>(); }
        
    void load(GeometryEncoder &encoder) override { encoder.instantiate(this, _reference.get()); }
    [[nodiscard]] bool is_instance() const noexcept override { return true; }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::ShapeInstance)
