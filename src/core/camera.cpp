//
// Created by Mike Smith on 2020/5/1.
//

#include "camera.h"

namespace luisa {

void Camera::generate_rays(KernelDispatcher &dispatch,
                           Sampler &sampler,
                           Viewport tile_viewport,
                           BufferView<float2> pixel_buffer,
                           BufferView<Ray> ray_buffer,
                           BufferView<float3> throughput_buffer) {
    
    if (auto filter = _film->filter(); filter == nullptr) {
        auto sample_buffer = sampler.generate_samples(dispatch, 2);
        dispatch(*_generate_pixel_samples_without_filter_kernel, tile_viewport.size.x * tile_viewport.size.y, [&](KernelArgumentEncoder &encode) {
            encode("sample_buffer", sample_buffer);
            encode("pixel_buffer", pixel_buffer);
            encode("throughput_buffer", throughput_buffer);
            encode("uniforms", camera::GeneratePixelSamplesWithoutFilterKernelUniforms{tile_viewport});
        });
    } else {
        _film->filter()->importance_sample_pixels(dispatch, tile_viewport, sampler, pixel_buffer, throughput_buffer);
    }
    _generate_rays(dispatch, sampler, tile_viewport, pixel_buffer, ray_buffer, throughput_buffer);
}

}