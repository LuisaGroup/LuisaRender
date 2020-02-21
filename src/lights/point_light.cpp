//
// Created by Mike Smith on 2020/2/17.
//

#include "point_light.h"

namespace luisa {

PointLight::PointLight(Device *device, const ParameterSet &parameter_set)
    : Light{device, parameter_set},
      _position{parameter_set["position"].parse_float3()},
      _emission{parameter_set["emission"].parse_float3_or_default(make_float3(parameter_set["emission"].parse_float()))} {}

std::unique_ptr<Kernel> PointLight::create_generate_samples_kernel() {
    return _device->create_kernel("point_light_generate_samples");
}

void PointLight::encode_data(TypelessBuffer &buffer, size_t index) {
    buffer.view_as<light::point::Data>(index)[0] = {_position, _emission};
}

size_t PointLight::data_stride() const noexcept {
    return sizeof(light::point::Data);
}

size_t PointLight::sample_dimensions() const noexcept {
    return 0;
}

uint PointLight::tag() const noexcept {
    static auto t = Light::_used_tag_count++;
    return t;
}

}
