//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include <core/ray.h>
#include <core/mathematics.h>

namespace luisa::light::point {

struct Data {
    float3 position;
    float3 emission;
};

LUISA_DEVICE_CALLABLE inline void generate_samples(
    LUISA_DEVICE_SPACE const uint *ray_queue,
    uint ray_queue_size,
    LUISA_DEVICE_SPACE const Data *light_buffer,
    LUISA_DEVICE_SPACE const uint *light_index_buffer,
    LUISA_DEVICE_SPACE const float3 *its_position_buffer,
    LUISA_DEVICE_SPACE float4 *Li_and_pdf_w_buffer,
    LUISA_DEVICE_SPACE bool *is_delta_buffer,
    LUISA_DEVICE_SPACE Ray *shadow_ray_buffer,
    uint tid) {
    
    if (tid < ray_queue_size) {
        auto ray_index = ray_queue[tid];
        auto light = light_buffer[light_index_buffer[tid]];
        auto its_p = its_position_buffer[ray_index];
        auto d = light.position - its_p;
        auto dd = max(1e-6f, dot(d, d));
    
        Li_and_pdf_w_buffer[ray_index] = make_float4(light.emission * (1.0f / dd), 1.0f);
        is_delta_buffer[ray_index] = true;
        
        auto distance = sqrt(dd);
        auto wo = d * (1.0f / distance);
        shadow_ray_buffer[ray_index] = make_ray(its_p, wo, 1e-4f, distance);
    }
    
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/light.h>

namespace luisa {

class PointLight : public Light {

protected:
    float3 _position;
    float3 _emission;

public:
    PointLight(Device *device, const ParameterSet &parameter_set);
    std::unique_ptr<Kernel> create_generate_samples_kernel() override;
    [[nodiscard]] size_t data_stride() const noexcept override;
    void encode_data(TypelessBuffer &buffer, size_t index) override;
    [[nodiscard]] size_t sample_dimensions() const noexcept override;
    [[nodiscard]] uint tag() const noexcept override;
};

}

#endif
