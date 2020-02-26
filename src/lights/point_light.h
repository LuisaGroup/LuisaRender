//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include <core/ray.h>
#include <core/selection.h>
#include <core/interaction.h>
#include <core/mathematics.h>

namespace luisa::light::point {

struct Data {
    float3 position;
    float3 emission;
};

LUISA_DEVICE_CALLABLE inline void generate_samples(
    LUISA_DEVICE_SPACE const Data *data_buffer,
    LUISA_DEVICE_SPACE const Selection *queue,
    uint queue_size,
    LUISA_DEVICE_SPACE uint8_t *its_state_buffer,
    LUISA_DEVICE_SPACE const float3 *its_position_buffer,
    LUISA_DEVICE_SPACE float4 *Li_and_pdf_w_buffer,
    LUISA_DEVICE_SPACE bool *is_delta_buffer,
    LUISA_DEVICE_SPACE Ray *shadow_ray_buffer,
    uint tid) {
    
    if (tid < queue_size) {
        auto selection = queue[tid];
        if (its_state_buffer[selection.interaction_index] & interaction_state_flags::VALID_BIT) {
            auto light_data = data_buffer[selection.data_index];
            auto its_p = its_position_buffer[selection.interaction_index];
            auto d = light_data.position - its_p;
            auto dd = max(1e-6f, dot(d, d));
            Li_and_pdf_w_buffer[selection.interaction_index] = make_float4(light_data.emission * (1.0f / dd), 1.0f);
            is_delta_buffer[selection.interaction_index] = true;
            auto distance = sqrt(dd);
            auto wo = d * (1.0f / distance);
            shadow_ray_buffer[selection.interaction_index] = make_ray(its_p, wo, 1e-4f, distance);
            its_state_buffer[selection.interaction_index] |= interaction_state_flags::DELTA_LIGHT_BIT;
        } else {
            shadow_ray_buffer[selection.interaction_index].max_distance = -1.0f;
        }
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
    void encode_data(TypelessBuffer &buffer, size_t data_index, uint2 cdf_range, uint instance_id, uint triangle_offset, uint vertex_offset, float shape_area) override;
    [[nodiscard]] uint tag() const noexcept override;
    [[nodiscard]] SampleLightsDispatch create_generate_samples_dispatch() override;
    [[nodiscard]] uint sampling_dimensions() const noexcept override;
};

}

#endif
