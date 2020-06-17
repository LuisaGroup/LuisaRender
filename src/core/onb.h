//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include <compute/data_types.h>
#include <compute/mathematics.h>

namespace luisa {

class Onb {

private:
    float3 _tangent{};
    float3 _binormal{};
    float3 _normal{};

public:
    LUISA_DEVICE_CALLABLE explicit Onb(float3 normal) noexcept : _normal{normal} {
        
        using namespace math;
        
        if (abs(_normal.x) > abs(_normal.z)) {
            _binormal.x = -_normal.y;
            _binormal.y = _normal.x;
            _binormal.z = 0;
        } else {
            _binormal.x = 0;
            _binormal.y = -_normal.z;
            _binormal.z = _normal.y;
        }
        _binormal = normalize(_binormal);
        _tangent = cross(_binormal, _normal);
    }
    
    [[nodiscard]] LUISA_DEVICE_CALLABLE float3 inverse_transform(float3 p) const noexcept {
        return p.x * _tangent + p.y * _binormal + p.z * _normal;
    }
    
    [[nodiscard]] LUISA_DEVICE_CALLABLE float3 transform(float3 p) const noexcept {
        using namespace math;
        return {dot(p, _tangent), dot(p, _binormal), dot(p, _normal)};
    }
    
};

}
