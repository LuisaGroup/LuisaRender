//
// Created by Mike Smith on 2020/2/9.
//

#pragma once

#include <core/transform.h>

namespace luisa {

class TRSTransform : public Transform {

protected:
    float3 _t;
    float4 _r;
    float3 _s;
    float4x4 _matrix{};

public:
    TRSTransform(Device *device, const ParameterSet &parameter_set);
    [[nodiscard]] float4x4 static_matrix() const noexcept override { return _matrix; }
    [[nodiscard]] auto translation() const noexcept { return _t; }
    [[nodiscard]] auto rotation() const noexcept { return _r; }
    [[nodiscard]] auto scaling() const noexcept { return _s; }
};

}
