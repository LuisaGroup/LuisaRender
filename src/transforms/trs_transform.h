//
// Created by Mike Smith on 2020/2/9.
//

#pragma once

#include <core/transform.h>

namespace luisa {

class TRSTransform : public Transform {

protected:
    float4x4 _matrix;

public:
    TRSTransform(Device *device, const ParameterSet &parameter_set);
    [[nodiscard]] float4x4 static_matrix() const noexcept override { return _matrix; }
};

LUISA_REGISTER_NODE_CREATOR("TRS", TRSTransform)

}
