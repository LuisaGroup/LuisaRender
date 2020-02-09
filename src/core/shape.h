//
// Created by Mike Smith on 2020/2/9.
//

#pragma once

#include "node.h"
#include "buffer.h"
#include "transform.h"
#include "material.h"
#include "geometry.h"

namespace luisa {

class Shape : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Shape);

protected:
    std::shared_ptr<Transform> _transform;
    std::shared_ptr<Material> _material;
    GeometryView _geometry_view{};

public:
    Shape(Device *device, const ParameterSet &parameter_set) : Node{device} {
        _transform = parameter_set["transform"].parse<Transform>();
        _material = parameter_set["material"].parse<Material>();
    }
    
    [[nodiscard]] Transform &transform() noexcept { return *_transform; }
    [[nodiscard]] Material &material() noexcept { return *_material; }
};

}
