//
// Created by Mike Smith on 2020/9/8.
//

#pragma once

#include "dual_buffer_variance.h"
#include "nlm_filter.h"
#include "gaussian_blur.h"

class FeaturePrefilter {

private:
    int _width;
    int _height;
    std::unique_ptr<DualBufferVariance> _dual_variance_stage;
    std::unique_ptr<NonLocalMeansFilter> _nlm_filter;
    std::unique_ptr<GaussianBlur> _gaussian_filter;
    KernelView _add_half_buffers_kernel;

public:
    FeaturePrefilter(Device &device,
                     TextureView feature, TextureView variance, TextureView feature_a, TextureView feature_b,
                     TextureView out_feature, TextureView out_var, TextureView out_a, TextureView out_b) noexcept
        : _width{static_cast<int>(feature.width())}, _height{static_cast<int>(feature.height())} {
        
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _dual_variance_stage = std::make_unique<DualBufferVariance>(device, 10, variance, feature_a, feature_b, out_var);
        _nlm_filter = std::make_unique<NonLocalMeansFilter>(device, 5, 3, 1.0f, feature, variance, feature_a, feature_b, out_a, out_b);
        _add_half_buffers_kernel = device.compile_kernel("feature_prefilter_add_half_buffers", [&] {
            auto p = thread_xy();
            If (p.x < _width && p.y < _height) {
                Var fa = make_float3(out_a.read(p));
                Var fb = make_float3(out_b.read(p));
                Var diff = fa - fb;
                out_feature.write(p, make_float4(0.5f * (fa + fb), 1.0f));
                out_var.write(p, make_float4(0.25f * diff * diff, 1.0f));
            };
        });
        _gaussian_filter = std::make_unique<GaussianBlur>(device, 0.5f, 0.5f, out_var, out_var);
    }
    
    void operator()(Dispatcher &dispatch) {
        dispatch(*_dual_variance_stage);
        dispatch(*_nlm_filter);
        dispatch(_add_half_buffers_kernel.parallelize(luisa::make_uint2(_width, _height)));
        dispatch(*_gaussian_filter);
    }
};
