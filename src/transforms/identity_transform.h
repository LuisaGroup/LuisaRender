//
// Created by Mike Smith on 2020/2/10.
//

#pragma once

#include <core/transform.h>

namespace luisa {

struct IdentityTransform : public Transform {
    IdentityTransform(Device *device, const ParameterSet &parameter_set) noexcept;
    [[nodiscard]] float4x4 matrix(float time[[maybe_unused]]) const override;
};

LUISA_REGISTER_NODE_CREATOR("Identity", IdentityTransform)

}
