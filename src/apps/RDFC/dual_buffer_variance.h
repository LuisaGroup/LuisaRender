//
// Created by Mike Smith on 2020/8/19.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>
#include <compute/kernel.h>

#include "box_blur.h"

using luisa::compute::Device;
using luisa::compute::KernelView;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class DualBufferVariance {

private:
    int _width;
    int _height;
    KernelView _dual_variance_kernel;
    KernelView _scale_kernel;
    TextureView _blurred_sample_variance;
    TextureView _blurred_dual_variance;
    std::unique_ptr<BoxBlur> _blur_sample_variance;
    std::unique_ptr<BoxBlur> _blur_dual_variance;

public:
    // Note: "var_sample" and "output" may alias
    DualBufferVariance(Device &device, int blur_radius, TextureView var_sample, TextureView color_a, TextureView color_b, TextureView output)
        : _width{static_cast<int>(var_sample.width())},
          _height{static_cast<int>(var_sample.height())},
          _blurred_sample_variance{device.allocate_texture<luisa::float4>(var_sample.width(), var_sample.height())},
          _blurred_dual_variance{device.allocate_texture<luisa::float4>(var_sample.width(), var_sample.height())} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _dual_variance_kernel = device.compile_kernel("dual_var", [&] {
            auto p = thread_xy();
            If (p.x < _width && p.y < _height) {
                Var va = color_a.read(p);
                Var vb = color_b.read(p);
                Var diff = va - vb;
                Var dual_var = 0.25f * diff * diff;
                _blurred_dual_variance.write(p, dual_var);
            };
        });
        
        _blur_dual_variance = std::make_unique<BoxBlur>(device, blur_radius, blur_radius, _blurred_dual_variance, _blurred_dual_variance);
        _blur_sample_variance = std::make_unique<BoxBlur>(device, blur_radius, blur_radius, var_sample, _blurred_sample_variance);
        
        _scale_kernel = device.compile_kernel("dual_var_scale", [&] {
            auto p = thread_xy();
            If (p.x < _width && p.y < _height) {
                Var sv = max(_blurred_sample_variance.read(p), 1e-6f);
                Var dv = _blurred_dual_variance.read(p);
                Var v = var_sample.read(p);
                output.write(p, min(v * dv / sv, 1e3f));
            };
        });
    }
    
    void operator()(Dispatcher &dispatch) {
        using namespace luisa;
        dispatch(_dual_variance_kernel.parallelize(make_uint2(_width, _height)));
        dispatch(*_blur_sample_variance);
        dispatch(*_blur_dual_variance);
        dispatch(_scale_kernel.parallelize(make_uint2(_width, _height)));
    }
};
