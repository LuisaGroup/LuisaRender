//
// Created by Mike Smith on 2020/2/10.
//

#pragma once

#include <core/shape.h>

namespace luisa {

class ShapeInstance : public Shape {

private:
    std::shared_ptr<Shape> _reference;

public:
    ShapeInstance(Device *device, const ParameterSet &parameter_set);
    void load(GeometryEncoder encoder) override;
    [[nodiscard]] bool is_instance() const noexcept override { return true; }
};

LUISA_REGISTER_NODE_CREATOR("Instance", ShapeInstance)

}
