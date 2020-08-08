//
// Created by Mike Smith on 2020/2/1.
//

#include <render/sampler.h>
#include <render/camera.h>
#include <core/mathematics.h>

#include "thin_lens.h"

namespace luisa {

class ThinLensCamera : public Camera {

protected:
    float3 _position{};
    float3 _front{};
    float3 _up{};
    float3 _left{};
    float _focal_plane_distance{};
    float _near_plane_distance{};
    float _lens_radius{};
    float2 _effective_sensor_size{};
    
    std::unique_ptr<Kernel> _generate_rays_kernel;

protected:
    void _generate_rays(KernelDispatcher &dispatch,
                        Sampler &sampler,
                        Viewport tile_viewport,
                        BufferView<float2> pixel_buffer,
                        BufferView<Ray> ray_buffer,
                        BufferView<float3> throughput_buffer) override;

public:
    ThinLensCamera(Device *device, const ParameterSet &parameters);
};

ThinLensCamera::ThinLensCamera(Device *device, const ParameterSet &parameters)
    : Camera{device, parameters} {
    
    auto sensor_size = 1e-3f * parameters["sensor_size"].parse_float2_or_default(make_float2(36.0f, 24.0f));
    auto film_resolution = make_float2(_film->resolution());
    auto film_aspect = film_resolution.x / film_resolution.y;
    auto sensor_aspect = sensor_size.x / sensor_size.y;
    if ((sensor_aspect < 1.0f && film_aspect > 1.0f) || (sensor_aspect > 1.0f && film_aspect < 1.0f)) {
        sensor_aspect = 1.0f / sensor_aspect;
        std::swap(sensor_size.x, sensor_size.y);
    }
    _effective_sensor_size = sensor_aspect < film_aspect ?
                             make_float2(sensor_size.x, sensor_size.x / film_aspect) :
                             make_float2(sensor_size.y * film_aspect, sensor_size.y);
    
    auto focal_length = 1e-3f * parameters["focal_length"].parse_float_or_default(50.0f);
    auto f_number = parameters["f_number"].parse_float_or_default(parameters["f_stop"].parse_float_or_default(1.2f));
    _lens_radius = 0.5f * focal_length / f_number;
    
    _position = parameters["position"].parse_float3();
    _up = parameters["up"].parse_float3_or_default(make_float3(0.0f, 1.0f, 0.0f));
    auto target = parameters["target"].parse_float3();
    auto forward = target - _position;
    _front = normalize(forward);
    _left = normalize(cross(_up, _front));
    _up = normalize(cross(_front, _left));
    
    _focal_plane_distance = length(forward);
    _near_plane_distance = 1.0f / (2.0f / focal_length - 1.0f / _focal_plane_distance);
    _generate_rays_kernel = device->load_kernel("camera::thin_lens::generate_rays");
}

void ThinLensCamera::_generate_rays(KernelDispatcher &dispatch,
                                   Sampler &sampler,
                                   Viewport tile_viewport,
                                   BufferView<float2> pixel_buffer,
                                   BufferView<Ray> ray_buffer,
                                   BufferView<float3> throughput_buffer) {
    
    auto pixel_count = tile_viewport.size.x * tile_viewport.size.y;
    auto sample_buffer = sampler.generate_samples(dispatch, 2);
    dispatch(*_generate_rays_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("ray_buffer", ray_buffer);
        encode("sample_buffer", sample_buffer);
        encode("ray_pixel_buffer", pixel_buffer);
        encode("uniforms", camera::thin_lens::GenerateRaysKernelUniforms{
            _position, _left, _up, _front,
            _film->resolution(), _effective_sensor_size,
            _near_plane_distance, _focal_plane_distance, _lens_radius,
            tile_viewport, _camera_to_world});
    });
}

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::ThinLensCamera)
