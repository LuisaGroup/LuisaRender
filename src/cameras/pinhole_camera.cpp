//
// Created by Mike Smith on 2020/2/20.
//

#include <core/mathematics.h>
#include "pinhole_camera.h"

namespace luisa {

void PinholeCamera::generate_rays(KernelDispatcher &dispatch,
                                  Sampler &sampler,
                                  Viewport tile_viewport,
                                  BufferView<float2> pixel_buffer,
                                  BufferView<Ray> ray_buffer,
                                  BufferView<float3> throughput_buffer) {
    
    auto pixel_count = tile_viewport.size.x * tile_viewport.size.y;
    auto sample_buffer = sampler.generate_camera_samples(dispatch);
    
    dispatch(*_generate_rays_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("sample_buffer", sample_buffer);
        encode("ray_pixel_buffer", pixel_buffer);
        encode("ray_buffer", ray_buffer);
        encode("ray_throughput_buffer", throughput_buffer);
        encode("uniforms", camera::pinhole::GenerateRaysKernelUniforms{
            _position, _left, _up, _front, _film->resolution(), _sensor_size, _near_plane, tile_viewport
        });
    });
}

PinholeCamera::PinholeCamera(Device *device, const ParameterSet &parameter_set)
    : Camera{device, parameter_set},
      _position{parameter_set["position"].parse_float3()},
      _near_plane{parameter_set["near_plane"].parse_float_or_default(0.1f)},
      _generate_rays_kernel{device->create_kernel("pinhole_camera_generate_rays")} {
    
    auto fov = radians(parameter_set["fov"].parse_float_or_default(35.0f));
    auto res = make_float2(_film->resolution());
    auto sensor_height = 2.0f * _near_plane * tan(0.5f * fov);
    auto sensor_width = sensor_height * res.x / res.y;
    _sensor_size = make_float2(sensor_width, sensor_height);
    
    auto up = parameter_set["up"].parse_float3_or_default(make_float3(0.0f, 1.0f, 0.0f));
    _front = normalize(parameter_set["target"].parse_float3() - _position);
    _left = normalize(cross(up, _front));
    _up = normalize(cross(_front, _left));
}

}
