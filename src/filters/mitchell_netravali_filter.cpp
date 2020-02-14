//
// Created by Mike Smith on 2020/2/2.
//

#include "mitchell_netravali_filter.h"

namespace luisa {

MitchellNetravaliFilter::MitchellNetravaliFilter(Device *device, const ParameterSet &parameters)
    : Filter{device, parameters},
      _b{parameters["b"].parse_float_or_default(1.0f / 3.0f)},
      _c{parameters["c"].parse_float_or_default(1.0f / 3.0f)} {
    
    _add_samples_kernel = device->create_kernel("mitchell_netravali_filter_add_samples");
}

void MitchellNetravaliFilter::add_samples(KernelDispatcher &dispatch,
                                          BufferView<float2> pixel_buffer,
                                          BufferView<float3> radiance_buffer,
                                          BufferView<uint> ray_queue_buffer,
                                          BufferView<uint> ray_queue_size_buffer,
                                          Film &film) {
    
    dispatch(*_add_samples_kernel, ray_queue_buffer.element_count(), [&](KernelArgumentEncoder &encode) {
        encode("ray_radiance_buffer", radiance_buffer);
        encode("ray_pixel_buffer", pixel_buffer);
        encode("ray_queue", ray_queue_buffer);
        encode("ray_queue_size", ray_queue_size_buffer);
        encode("accumulation_buffer", film.accumulation_buffer());
        encode("uniforms", MitchellNetravaliFilterAddSamplesKernelUniforms{film.resolution(), _radius, _b, _c});
    });
}

}
