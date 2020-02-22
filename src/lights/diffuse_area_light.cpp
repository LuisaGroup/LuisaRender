//
// Created by Mike Smith on 2020/2/22.
//

#include "diffuse_area_light.h"

namespace luisa {

uint DiffuseAreaLight::tag() const noexcept {
    static auto t = Light::_assign_tag();
    return t;
}

std::unique_ptr<Kernel> DiffuseAreaLight::create_generate_samples_kernel() {
    return std::unique_ptr<Kernel>();
}

Light::SampleLightsDispatch DiffuseAreaLight::create_generate_samples_dispatch() {
    return luisa::Light::SampleLightsDispatch();
}

size_t DiffuseAreaLight::data_stride() const noexcept {
    return sizeof(light::diffuse_area::Data);
}

Shape *DiffuseAreaLight::shape() const noexcept {
    return _shape.get();
}

uint DiffuseAreaLight::sampling_dimensions() const noexcept {
    return 2u;
}

void DiffuseAreaLight::encode_data(TypelessBuffer &buffer, size_t index) {

}

DiffuseAreaLight::DiffuseAreaLight(Device *device, const ParameterSet &parameter_set)
    : Light{device, parameter_set}, _shape{parameter_set["shape"].parse<Shape>()} {}
    
}
