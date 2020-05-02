//
// Created by Mike Smith on 2020/2/9.
//

#pragma once

#include "node.h"
#include "buffer.h"
#include "transform.h"
#include "material.h"
#include "geometry.h"
#include "light.h"

namespace luisa {

class Shape : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Shape);

protected:
    std::shared_ptr<Transform> _transform;
    std::shared_ptr<Material> _material;

public:
    Shape(Device *device, const ParameterSet &parameter_set)
        : Node{device},
          _transform{parameter_set["transform"].parse_or_null<Transform>()},
          _material{parameter_set["material"].parse_or_null<Material>()} {
        
        if (_transform == nullptr) {
            LUISA_WARNING("No transform specific for shape, using IdentityTransform by default");
            _transform = std::make_shared<IdentityTransform>(_device);
        }
    }
    
    [[nodiscard]] Transform &transform() noexcept { return *_transform; }
    [[nodiscard]] Material *material() noexcept { return _material.get(); }
    virtual void load(GeometryEncoder &encoder) = 0;
    [[nodiscard]] virtual bool is_instance() const noexcept { return false; }
};

}
