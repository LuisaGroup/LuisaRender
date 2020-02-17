//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include <core/data_types.h>

namespace luisa::point_light {

struct Data {
    float3 position;
    float3 emission;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/light.h>

namespace luisa {

class PointLight : public Light {

protected:
    LUISA_MAKE_LIGHT_TAG_ASSIGNMENT("Point")

protected:
    float3 _position;
    float3 _emission;

public:
    PointLight(Device *device, const ParameterSet &parameter_set);
    std::unique_ptr<Kernel> create_sample_kernel() override;
    [[nodiscard]] size_t data_stride() const noexcept override;
    void encode_data(TypelessBuffer &buffer, size_t index) override;
};

LUISA_REGISTER_NODE_CREATOR("Point", PointLight)

}

#endif
