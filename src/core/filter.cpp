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
        encode("uniforms", filter::separable::ImportanceSamplePixelsKernelUniforms{tile_viewport, _radius});
    });
}

SeparableFilter::SeparableFilter(Device *device, const ParameterSet &parameters)
    : Filter{device, parameters},
      _importance_sample_pixels_kernel{device->create_kernel("separable_filter_importance_sample_pixels")} {}

void SeparableFilter::_compute_lut_if_necessary() {
    
    if (!_lut_computed) {
        
        for (auto i = 1u; i <= filter::separable::WEIGHT_TABLE_SIZE; i++) {
            auto offset = (static_cast<float>(i) / static_cast<float>(filter::separable::WEIGHT_TABLE_SIZE) * 2.0f - 1.0f) / _radius;
            _lut.w[i] = _weight(offset);
        }
        for (auto i = 0u; i < filter::separable::CDF_TABLE_SIZE; i++) {
            auto offset = (static_cast<float>(i) / static_cast<float>(filter::separable::CDF_TABLE_SIZE) * 2.0f - 1.0f) / _radius;
            _lut.cdf[i] = std::abs(_weight(offset));
        }
        auto sum = 0.0f;
        for (float &cdf : _lut.cdf) {
            sum += cdf;
            cdf = sum;
        }
        
        auto inv_sum = 1.0f / sum;
        for (float &cdf : _lut.cdf) { cdf *= inv_sum; }
    }
    
    _lut_computed = true;
}
    
}