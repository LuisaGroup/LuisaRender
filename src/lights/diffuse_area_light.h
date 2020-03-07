//
// Created by Mike Smith on 2020/2/22.
//

#pragma once

#include <core/mathematics.h>
#include <core/ray.h>
#include <core/interaction.h>
#include <core/sampling.h>
#include <core/light.h>

namespace luisa::light::diffuse_area {

struct Data {
    float3 emission;
    uint2 cdf_range;
    uint instance_id;
    uint triangle_offset;
    uint vertex_offset;
    float shape_area;
    bool two_sided;
};

static_assert(sizeof(Data) == 48ul);

LUISA_DEVICE_CALLABLE inline void generate_samples(
    LUISA_DEVICE_SPACE const Data *data_buffer,
    LUISA_DEVICE_SPACE const float3 *sample_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE const float *cdf_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const light::Selection *queue,
    uint queue_size,
    LUISA_DEVICE_SPACE const uint8_t *its_state_buffer,
    LUISA_DEVICE_SPACE const float3 *its_position_buffer,
    LUISA_DEVICE_SPACE float4 *Li_and_pdf_w_buffer,
    LUISA_DEVICE_SPACE Ray *shadow_ray_buffer,
    uint tid) {
    
    if (tid < queue_size) {
        
        auto selection = queue[tid];
        
        if (its_state_buffer[selection.interaction_index] & interaction::state::HIT) {
            auto light_data = data_buffer[selection.data_index];
            auto r = sample_buffer[tid];
            auto cdf_offset = sample_discrete(cdf_buffer, light_data.cdf_range.x, light_data.cdf_range.y, r.x) - light_data.cdf_range.x;
            auto indices = index_buffer[light_data.triangle_offset + cdf_offset] + light_data.vertex_offset;
            auto b = uniform_sample_triangle(r.y, r.z);
            auto p_entity = b.x * position_buffer[indices.x] + b.y * position_buffer[indices.y] + (1.0f - b.x - b.y) * position_buffer[indices.z];
            auto n_entity = b.x * normal_buffer[indices.x] + b.y * normal_buffer[indices.y] + (1.0f - b.x - b.y) * normal_buffer[indices.z];
            auto transform = transform_buffer[light_data.instance_id];
            auto p_light = make_float3(transform * make_float4(p_entity, 1.0f));
            auto n_light = normalize(transpose(inverse(make_float3x3(transform))) * n_entity);
            auto p_hit = its_position_buffer[selection.interaction_index];
            auto L = p_light - p_hit;
            auto wi = normalize(L);
            auto cos_theta = dot(-wi, n_light);
            auto distance = length(L);
            auto pdf_w = distance * distance / (light_data.shape_area * max(abs(cos_theta), 1e-4f));
            if (light_data.two_sided || cos_theta > 1e-4f) {
                Li_and_pdf_w_buffer[selection.interaction_index] = make_float4(light_data.emission, pdf_w);
                shadow_ray_buffer[selection.interaction_index] = make_ray(p_hit, wi, 1e-4f, distance - 1e-4f);
            } else {
                Li_and_pdf_w_buffer[selection.interaction_index] = make_float4(make_float3(), pdf_w);
                shadow_ray_buffer[selection.interaction_index].max_distance = -1.0f;
            }
        } else {
            shadow_ray_buffer[selection.interaction_index].max_distance = -1.0f;
        }
    }
}

LUISA_DEVICE_CALLABLE inline void evaluate_emissions(
    LUISA_DEVICE_SPACE const Data *data_buffer,
    LUISA_DEVICE_SPACE const light::Selection *queue,
    uint queue_size,
    LUISA_DEVICE_SPACE const float3 *its_normal_buffer,
    LUISA_DEVICE_SPACE const float4 *its_wo_and_distance_buffer,
    LUISA_DEVICE_SPACE float4 *its_emission_and_pdf_buffer,
    uint tid) {
    
    if (tid < queue_size) {
        auto selection = queue[tid];
        auto light_data = data_buffer[selection.data_index];
        auto normal = its_normal_buffer[selection.interaction_index];
        auto wo_and_distance = its_wo_and_distance_buffer[selection.interaction_index];
        auto wo = make_float3(wo_and_distance);
        auto distance = max(wo_and_distance.w, 1e-3f);
        auto cos_theta = dot(normal, wo);
        auto emission = (cos_theta > 1e-4f || light_data.two_sided) ? light_data.emission : make_float3();
        auto pdf = abs(cos_theta) / (distance * distance);
        its_emission_and_pdf_buffer[selection.interaction_index] = make_float4(emission, pdf);
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/shape.h>

namespace luisa {

class DiffuseAreaLight : public Light {

protected:
    float3 _emission;
    std::shared_ptr<Shape> _shape;
    bool _two_sided;

public:
    DiffuseAreaLight(Device *device, const ParameterSet &parameter_set);
    [[nodiscard]] uint tag() const noexcept override;
    [[nodiscard]] std::unique_ptr<Kernel> create_generate_samples_kernel() override;
    [[nodiscard]] SampleLightsDispatch create_generate_samples_dispatch() override;
    std::unique_ptr<Kernel> create_evaluate_emissions_kernel() override;
    EvaluateLightsDispatch create_evaluate_emissions_dispatch() override;
    [[nodiscard]] size_t data_stride() const noexcept override;
    [[nodiscard]] Shape *shape() const noexcept override;
    [[nodiscard]] uint sampling_dimensions() const noexcept override;
    void encode_data(TypelessBuffer &buffer, size_t data_index, uint2 cdf_range, uint instance_id, uint triangle_offset, uint vertex_offset, float shape_area) override;
};

}

#endif
