//
// Created by Mike Smith on 2020/2/22.
//

#pragma once

namespace luisa::light::diffuse_area {

struct Data {

};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/shape.h>
#include <core/light.h>

namespace luisa {

class DiffuseAreaLight : public Light {

protected:
    std::shared_ptr<Shape> _shape;

public:
    DiffuseAreaLight(Device *device, const ParameterSet &parameter_set);
    [[nodiscard]] uint tag() const noexcept override;
    [[nodiscard]] std::unique_ptr<Kernel> create_generate_samples_kernel() override;
    [[nodiscard]] SampleLightsDispatch create_generate_samples_dispatch() override;
    [[nodiscard]] size_t data_stride() const noexcept override;
    [[nodiscard]] Shape *shape() const noexcept override;
    [[nodiscard]] uint sampling_dimensions() const noexcept override;
    void encode_data(TypelessBuffer &buffer, size_t index) override;
};

}

#endif
