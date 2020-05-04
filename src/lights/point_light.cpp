//
// Created by Mike Smith on 2020/2/17.
//

#include "point_light.h"

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

LUISA_REGISTER_NODE_CREATOR("Point", PointLight)

PointLight::PointLight(Device *device, const ParameterSet &parameter_set)
    : Light{device, parameter_set},
      _position{parameter_set["position"].parse_float3()},
      _emission{parameter_set["emission"].parse_float3_or_default(make_float3(parameter_set["emission"].parse_float()))} {}

std::unique_ptr<Kernel> PointLight::create_generate_samples_kernel() {
    return _device->load_kernel("point_light_generate_samples");
}

void PointLight::encode_data(TypelessBuffer &buffer, size_t data_index, uint2, uint, uint, uint, float) {
    buffer.view_as<light::point::Data>(data_index)[0] = {_position, _emission};
}

size_t PointLight::data_stride() const noexcept {
    return sizeof(light::point::Data);
}

uint PointLight::tag() const noexcept {
    static auto t = Light::_assign_tag();
    return t;
}

Light::SampleLightsDispatch PointLight::create_generate_samples_dispatch() {
    
    return [](KernelDispatcher &dispatch, Kernel &kernel, uint dispatch_extent, BufferView<float>,
              TypelessBuffer &light_data_buffer, BufferView<light::Selection> queue, BufferView<uint> queue_size,
              BufferView<float>, InteractionBufferSet &interactions, Geometry *, LightSampleBufferSet &light_samples) {
        
        dispatch(kernel, dispatch_extent, [&](KernelArgumentEncoder &encode) {
            encode("data_buffer", light_data_buffer.view_as<light::point::Data>());
            encode("queue", queue);
            encode("queue_size", queue_size);
            encode("its_state_buffer", interactions.state_buffer());
            encode("its_position_buffer", interactions.position_buffer());
            encode("Li_and_pdf_w_buffer", light_samples.radiance_and_pdf_w_buffer());
            encode("shadow_ray_buffer", light_samples.shadow_ray_buffer());
        });
    };
}

uint PointLight::sampling_dimensions() const noexcept {
    return 0;
}

}
