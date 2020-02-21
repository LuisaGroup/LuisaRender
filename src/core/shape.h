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
    static constexpr auto INVALID_ENTITY_INDEX = std::numeric_limits<uint>::max();
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Shape);

protected:
    std::shared_ptr<Transform> _transform;
    std::shared_ptr<Material> _material;
    std::shared_ptr<Light> _emission;
    uint _entity_index{INVALID_ENTITY_INDEX};

public:
    Shape(Device *device, const ParameterSet &parameter_set)
        : Node{device},
          _transform{parameter_set["transform"].parse_or_null<Transform>()},
          _material{parameter_set["material"].parse_or_null<Material>()},
          _emission{parameter_set["emission"].parse_or_null<Light>()} {
        
        LUISA_ERROR_IF_NOT(_emission == nullptr || _emission->is_shape_applicable(), "light source not applicable to shape");
    }
    
    [[nodiscard]] Transform *transform() noexcept { return _transform.get(); }
    [[nodiscard]] Material *material() noexcept { return _material.get(); }
    [[nodiscard]] Light *emission() const noexcept { return _emission.get(); }
    
    [[nodiscard]] bool loaded() const noexcept { return _entity_index != INVALID_ENTITY_INDEX; }
    [[nodiscard]] uint entity_index() const {
        LUISA_ERROR_IF_NOT(loaded(), "shape not loaded");
        return _entity_index;
    }
    
    virtual void load(GeometryEncoder &encoder) = 0;
    [[nodiscard]] virtual bool is_instance() const noexcept { return false; }
};

}
