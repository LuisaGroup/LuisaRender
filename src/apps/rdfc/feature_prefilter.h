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
    std::shared_ptr<Kernel> _add_half_buffers_kernel;

public:
    FeaturePrefilter(Device &device,
                     Texture &feature, Texture &variance, Texture &feature_a, Texture &feature_b,
                     Texture &out_feature, Texture &out_var, Texture &out_a, Texture &out_b) noexcept
        : _width{static_cast<int>(feature.width())}, _height{static_cast<int>(feature.height())} {
        
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _dual_variance_stage = std::make_unique<DualBufferVariance>(device, 10, variance, feature_a, feature_b, out_var);
        _nlm_filter = std::make_unique<NonLocalMeansFilter>(device, 5, 3, 1.0f, feature, variance, feature_a, feature_b, out_a, out_b);
        _add_half_buffers_kernel = device.compile_kernel("feature_prefilter_add_half_buffers", [&] {
            Arg<ReadOnlyTex2D> feature_a_flt{out_a};
            Arg<ReadOnlyTex2D> feature_b_flt{out_b};
            Arg<WriteOnlyTex2D> feature_flt{out_feature};
            Arg<WriteOnlyTex2D> residual_var{out_var};
            auto p = thread_xy();
            If (p.x() < _width && p.y() < _height) {
                Auto fa = make_float3(read(feature_a_flt, p));
                Auto fb = make_float3(read(feature_b_flt, p));
                Auto diff = fa - fb;
                write(feature_flt, p, make_float4(0.5f * (fa + fb), 1.0f));
                write(residual_var, p, make_float4(0.25f * diff * diff, 1.0f));
            };
        });
        _gaussian_filter = std::make_unique<GaussianBlur>(device, 0.5f, 0.5f, out_var, out_var);
    }
    
    void operator()(Dispatcher &dispatch) {
        dispatch(*_dual_variance_stage);
        dispatch(*_nlm_filter);
        dispatch(_add_half_buffers_kernel->parallelize(luisa::make_uint2(_width, _height)));
        dispatch(*_gaussian_filter);
    }
};
