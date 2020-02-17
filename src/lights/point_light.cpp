//
// Created by Mike Smith on 2020/2/17.
//

#include "point_light.h"

namespace luisa {

PointLight::PointLight(Device *device, const ParameterSet &parameter_set)
    : Light{device, parameter_set},
      _position{parameter_set["position"].parse_float3()},
      _emission{parameter_set["emission"].parse_float3()} {}

std::unique_ptr<Kernel> PointLight::create_sample_kernel() {
    return _device->create_kernel("point_light_sample");
}

void PointLight::encode_data(TypelessBuffer &buffer, size_t index) {
    buffer.view_as<point_light::Data>(index)[0] = {_position, _emission};
}

size_t PointLight::data_stride() const noexcept {
    return sizeof(point_light::Data);
}

}
