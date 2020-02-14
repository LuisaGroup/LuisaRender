//
// Created by Mike Smith on 2020/2/1.
//

#include <core/sampler.h>
#include "thin_lens_camera.h"

namespace luisa {

ThinLensCamera::ThinLensCamera(Device *device, const ParameterSet &parameters)
    : Camera{device, parameters} {
    
    _sensor_size = 1e-3f * parameters["sensor_size"].parse_float2_or_default(make_float2(36.0f, 24.0f));
    auto film_resolution = make_float2(_film->resolution());
    auto film_aspect = film_resolution.x / film_resolution.y;
    auto sensor_aspect = _sensor_size.x / _sensor_size.y;
    _effective_sensor_size = sensor_aspect < film_aspect ?
                             make_float2(_sensor_size.x, _sensor_size.x / film_aspect) :
                             make_float2(_sensor_size.y * film_aspect, _sensor_size.y);
    
    auto focal_length = 1e-3f * parameters["focal_length"].parse_float_or_default(50.0f);
    auto f_number = parameters["f_number"].parse_float_or_default(1.2f);
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
    _generate_rays_kernel = device->create_kernel("thin_lens_camera_generate_rays");
}

void ThinLensCamera::generate_rays(KernelDispatcher &dispatch,
                                   BufferView<float4> sample_buffer,
                                   BufferView<float2> pixel_buffer,
                                   BufferView<Ray> ray_buffer,
                                   BufferView<float3> throughput_buffer) {
    
    dispatch(*_generate_rays_kernel, _film->resolution(), [&](KernelArgumentEncoder &encode) {
        encode("ray_buffer", ray_buffer);
        encode("ray_throughput_buffer", throughput_buffer);
        encode("sample_buffer", sample_buffer);
        encode("ray_pixel_buffer", pixel_buffer);
        encode("uniforms", thin_lens_camera::GenerateRaysKernelUniforms{
            _position, _left, _up, _front,
            _film->resolution(), _effective_sensor_size,
            _near_plane_distance, _focal_plane_distance, _lens_radius});
    });
}

}
