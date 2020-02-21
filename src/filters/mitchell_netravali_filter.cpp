//
// Created by Mike Smith on 2020/2/2.
//

#include "mitchell_netravali_filter.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("MitchellNetravali", MitchellNetravaliFilter)

MitchellNetravaliFilter::MitchellNetravaliFilter(Device *device, const ParameterSet &parameters)
    : Filter{device, parameters},
      _b{parameters["b"].parse_float_or_default(1.0f / 3.0f)},
      _c{parameters["c"].parse_float_or_default(1.0f / 3.0f)} {
    
    _apply_and_accumulate_kernel = device->create_kernel("mitchell_netravali_filter_apply_and_accumulate");
}

void MitchellNetravaliFilter::apply_and_accumulate(KernelDispatcher &dispatch,
                                                   uint2 film_resolution,
                                                   Viewport film_viewport,
                                                   Viewport tile_viewport,
                                                   BufferView<float2> pixel_buffer,
                                                   BufferView<float3> color_buffer,
                                                   BufferView<float4> accumulation_buffer) {
    
    dispatch(*_apply_and_accumulate_kernel, tile_viewport.size.x * tile_viewport.size.y, [&](KernelArgumentEncoder &encode) {
        encode("ray_color_buffer", color_buffer);
        encode("ray_pixel_buffer", pixel_buffer);
        encode("accumulation_buffer", accumulation_buffer);
        encode("uniforms", filter::mitchell_netravali::ApplyAndAccumulateKernelUniforms{
            _filter_viewport(film_viewport, tile_viewport), tile_viewport, film_resolution, _radius, _b, _c});
    });
}

}
