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
    static constexpr auto INVALID_ENTITY_INDEX = std::numeric_limits<uint>::max();
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Shape);

protected:
    std::shared_ptr<Transform> _transform;
    std::shared_ptr<Material> _material;
    uint _entity_index{INVALID_ENTITY_INDEX};

public:
    Shape(Device *device, const ParameterSet &parameter_set) : Node{device} {
        _material = parameter_set["material"].parse_or_null<Material>();
        _transform = parameter_set["transform"].parse_or_null<Transform>();
        if (_transform == nullptr) {
            _transform = std::make_shared<Transform>(_device);
        }
    }
    
    [[nodiscard]] Transform &transform() noexcept { return *_transform; }
    [[nodiscard]] Material &material() noexcept { return *_material; }
    [[nodiscard]] bool loaded() const noexcept { return _entity_index != INVALID_ENTITY_INDEX; }
    [[nodiscard]] uint entity_index() const {
        LUISA_ERROR_IF_NOT(loaded(), "shape not loaded");
        return _entity_index;
    }
    
    virtual void load(GeometryEncoder &encoder) = 0;
    [[nodiscard]] virtual bool is_instance() const noexcept { return false; }
};

}
