//
// Created by Mike Smith on 2020/2/2.
//

#include "mitchell_netravali_filter.h"

namespace luisa {

void MitchellNetravaliFilter::add_samples(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue, Film &film) {
    dispatch(*_add_samples_kernel, ray_queue.capacity(), [&](KernelArgumentEncoder &encode) {
        encode("ray_radiance_buffer", ray_pool.attribute_buffer<float3>("radiance"));
        encode("ray_pixel_buffer", ray_pool.attribute_buffer<float2>("pixel"));
        encode("ray_queue", ray_queue.index_buffer);
        encode("ray_queue_size", ray_queue.size_buffer);
        encode("accumulation_buffer", film.accumulation_buffer());
        encode("uniforms", MitchellNetravaliFilterAddSamplesKernelUniforms{film.resolution(), _a, _b, _c});
    });
}

}