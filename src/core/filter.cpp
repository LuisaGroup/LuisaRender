//
// Created by Mike Smith on 2020/5/1.
//

#include "filter.h"

namespace luisa {

void SeparableFilter::importance_sample_pixels(KernelDispatcher &dispatch,
                                               Viewport tile_viewport,
                                               Sampler &sampler,
                                               BufferView<float2> pixel_location_buffer,
                                               BufferView<float3> pixel_weight_buffer) {
    
    _compute_lut_if_necessary();
    auto sample_buffer = sampler.generate_samples(dispatch, 2);
    dispatch(*_importance_sample_pixels_kernel, tile_viewport.size.x * tile_viewport.size.y, [&](KernelArgumentEncoder &encode) {
        encode("lut", _lut);
        encode("random_buffer", sample_buffer);
        encode("pixel_location_buffer", pixel_location_buffer);
        encode("pixel_weight_buffer", pixel_weight_buffer);
        encode("uniforms", filter::separable::ImportanceSamplePixelsKernelUniforms{tile_viewport, _radius, _scale});
    });
}

SeparableFilter::SeparableFilter(Device *device, const ParameterSet &parameters)
    : Filter{device, parameters},
      _importance_sample_pixels_kernel{device->load_kernel("separable_filter::importance_sample_pixels")} {}

void SeparableFilter::_compute_lut_if_necessary() {
    
    if (!_lut_computed) {
        
        constexpr auto inv_table_size = 1.0f / static_cast<float>(filter::separable::TABLE_SIZE);
        
        auto abs_sum = 0.0f;
        for (auto i = 0u; i < filter::separable::TABLE_SIZE; i++) {
            auto offset = (static_cast<float>(i) * inv_table_size * 2.0f - 1.0f) * _radius;
            auto w = _weight_1d(offset);
            _lut.w[i] = w;
            _lut.cdf[i] = (abs_sum += std::abs(w));
        }
        auto inv_sum = 1.0f / abs_sum;
        for (float &cdf : _lut.cdf) { cdf *= inv_sum; }
        
        auto absolute_volume = 0.0f;
        auto signed_volume = 0.0f;
        for (float u : _lut.w) {
            for (float v : _lut.w) {
                signed_volume += u * v;
                absolute_volume += std::abs(u * v);
            }
        }
        _scale = absolute_volume / signed_volume;
        
        _lut_computed = true;
    }
}
    
}
