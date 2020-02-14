//
// Created by Mike Smith on 2020/2/2.
//

#include "mitchell_netravali_filter.h"

namespace luisa {

MitchellNetravaliFilter::MitchellNetravaliFilter(Device *device, const ParameterSet &parameters)
    : Filter{device, parameters},
      _b{parameters["b"].parse_float_or_default(1.0f / 3.0f)},
      _c{parameters["c"].parse_float_or_default(1.0f / 3.0f)} {
    
    _apply_kernel = device->create_kernel("mitchell_netravali_filter_apply");
}

void MitchellNetravaliFilter::apply(KernelDispatcher &dispatch,
                                    BufferView<float2> pixel_buffer,
                                    BufferView<float3> radiance_buffer,
                                    BufferView<float4> frame,
                                    uint2 film_resolution) {
    
    dispatch(*_apply_kernel, film_resolution, [&](KernelArgumentEncoder &encode) {
        encode("ray_radiance_buffer", radiance_buffer);
        encode("ray_pixel_buffer", pixel_buffer);
        encode("frame", frame);
        encode("uniforms", mitchell_netravali_filter::ApplyKernelUniforms{film_resolution, _radius, _b, _c});
    });
}

}
