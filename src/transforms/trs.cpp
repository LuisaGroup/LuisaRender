//
// Created by Mike Smith on 2020/9/14.
//

#include <render/parser.h>
#include <render/transform.h>

namespace luisa::render::transform {

class TRSTransform : public Transform {

protected:
    float3 _t;
    float4 _r;
    float3 _s;
    float4x4 _matrix{};

public:
    TRSTransform(Device *device, const ParameterSet &parameter_set)
        : Transform{device, parameter_set},
          _t{parameter_set["translation"].parse_float3_or_default(make_float3(0.0f))},
          _r{parameter_set["rotation"].parse_float4_or_default(make_float4(0.0f, 1.0f, 0.0f, 0.0f))},
          _s{parameter_set["scaling"].parse_float3_or_default(make_float3(parameter_set["scaling"].parse_float_or_default(1.0f)))} {
    
        _r.w = math::radians(_r.w);
        _matrix = math::translation(_t) * math::rotation(make_float3(_r), _r.w) * math::scaling(_s);
    }
    
    [[nodiscard]] float4x4 matrix(float time) const noexcept override { return _matrix; }
    [[nodiscard]] auto translation() const noexcept { return _t; }
    [[nodiscard]] auto rotation() const noexcept { return _r; }
    [[nodiscard]] auto scaling() const noexcept { return _s; }
    [[nodiscard]] bool is_static() const noexcept override { return true; }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::transform::TRSTransform)
