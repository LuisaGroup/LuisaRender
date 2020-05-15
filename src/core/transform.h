//
// Created by Mike Smith on 2020/2/9.
//

#pragma once

#include "plugin.h"
#include "parser.h"
#include "mathematics.h"

namespace luisa {

class Transform : public Plugin {

protected:
    explicit Transform(Device *device) : Plugin{device} {}

public:
    Transform(Device *device, const ParameterSet &) : Plugin{device} {}
    [[nodiscard]] virtual float4x4 static_matrix() const { return math::identity(); }
    [[nodiscard]] virtual float4x4 dynamic_matrix(float time[[maybe_unused]]) const {
        LUISA_EXCEPTION_IF_NOT(is_static(), "Transform::dynamic_matrix() not implemented in dynamic transform");
        return math::identity();
    }
    [[nodiscard]] virtual bool is_static() const noexcept { return true; }
};

class IdentityTransform : public Transform {

public:
    explicit IdentityTransform(Device *device) noexcept : Transform{device};
};

}
