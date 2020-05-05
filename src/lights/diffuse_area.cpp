//
// Created by Mike Smith on 2020/2/22.
//

#include <core/shape.h>

#include "diffuse_area.h"

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

LUISA_REGISTER_NODE_CREATOR("DiffuseArea", DiffuseAreaLight)

uint DiffuseAreaLight::tag() const noexcept {
    static auto t = Light::_assign_tag();
    return t;
}

std::unique_ptr<Kernel> DiffuseAreaLight::create_generate_samples_kernel() {
    return _device->load_kernel("diffuse_area_light::generate_samples");
}

Light::SampleLightsDispatch DiffuseAreaLight::create_generate_samples_dispatch() {
    return [](KernelDispatcher &dispatch, Kernel &kernel, uint dispatch_extent, BufferView<float> sample_buffer,
              TypelessBuffer &light_data_buffer, BufferView<light::Selection> queue, BufferView<uint> queue_size,
              BufferView<float> cdf_buffer,
              InteractionBufferSet &interactions, Geometry *geometry, LightSampleBufferSet &light_samples) {
        
        dispatch(kernel, dispatch_extent, [&](KernelArgumentEncoder &encode) {
            encode("data_buffer", light_data_buffer.view_as<light::diffuse_area::Data>());
            encode("sample_buffer", sample_buffer);
            encode("transform_buffer", geometry->transform_buffer());
            encode("cdf_buffer", cdf_buffer);
            encode("index_buffer", geometry->index_buffer());
            encode("position_buffer", geometry->position_buffer());
            encode("normal_buffer", geometry->normal_buffer());
            encode("queue", queue);
            encode("queue_size", queue_size);
            encode("its_state_buffer", interactions.state_buffer());
            encode("its_position_buffer", interactions.position_buffer());
            encode("Li_and_pdf_w_buffer", light_samples.radiance_and_pdf_w_buffer());
            encode("shadow_ray_buffer", light_samples.shadow_ray_buffer());
        });
    };
}

size_t DiffuseAreaLight::data_stride() const noexcept {
    return sizeof(light::diffuse_area::Data);
}

Shape *DiffuseAreaLight::shape() const noexcept {
    return _shape.get();
}

uint DiffuseAreaLight::sampling_dimensions() const noexcept {
    return 3u;
}

void DiffuseAreaLight::encode_data(TypelessBuffer &buffer, size_t data_index, uint2 cdf_range, uint instance_id, uint triangle_offset, uint vertex_offset, float shape_area) {
    buffer.view_as<light::diffuse_area::Data>(data_index)[0] = {_emission, cdf_range, instance_id, triangle_offset, vertex_offset, shape_area, _two_sided};
}

DiffuseAreaLight::DiffuseAreaLight(Device *device, const ParameterSet &parameter_set)
    : Light{device, parameter_set},
      _emission{parameter_set["emission"].parse_float3_or_default(make_float3(parameter_set["emission"].parse_float()))},
      _shape{parameter_set["shape"].parse<Shape>()},
      _two_sided{parameter_set["two_sided"].parse_bool_or_default(false)} {}

std::unique_ptr<Kernel> DiffuseAreaLight::create_evaluate_emissions_kernel() {
    return _device->load_kernel("diffuse_area::light_evaluate_emissions");
}

Light::EvaluateLightsDispatch DiffuseAreaLight::create_evaluate_emissions_dispatch() {
    return [](KernelDispatcher &dispatch, Kernel &kernel, uint dispatch_extent,
              TypelessBuffer &light_data_buffer, BufferView<light::Selection> queue, BufferView<uint> queue_size,
              InteractionBufferSet &interactions) {
        
        dispatch(kernel, dispatch_extent, [&](KernelArgumentEncoder &encode) {
            encode("data_buffer", light_data_buffer);
            encode("queue", queue);
            encode("queue_size", queue_size);
            encode("its_emission_buffer", interactions.emission_buffer());
            encode("its_state_buffer", interactions.state_buffer());
        });
    };
}
    
}
