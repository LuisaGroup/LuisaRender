//
// Created by Mike Smith on 2020/8/19.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>
#include <compute/kernel.h>

#include "box_blur.h"

using luisa::compute::Device;
using luisa::compute::Kernel;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class DualBufferVariance {

private:
    int _width;
    int _height;
    std::unique_ptr<Kernel> _dual_variance_kernel;
    std::unique_ptr<Kernel> _scale_kernel;
    std::unique_ptr<Texture> _blurred_sample_variance;
    std::unique_ptr<Texture> _blurred_dual_variance;
    std::unique_ptr<BoxBlur> _blur_sample_variance;
    std::unique_ptr<BoxBlur> _blur_dual_variance;

public:
    // Note: "var_sample" and "output" may alias
    DualBufferVariance(Device &device, int blur_radius, Texture &var_sample, Texture &color_a, Texture &color_b, Texture &output)
        : _width{static_cast<int>(var_sample.width())},
          _height{static_cast<int>(var_sample.height())},
          _blurred_sample_variance{device.allocate_texture<luisa::float4>(var_sample.width(), var_sample.height())},
          _blurred_dual_variance{device.allocate_texture<luisa::float4>(var_sample.width(), var_sample.height())} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _dual_variance_kernel = device.compile_kernel([&] {
            
            Arg<ReadOnlyTex2D> color_a_texture{color_a};
            Arg<ReadOnlyTex2D> color_b_texture{color_b};
            Arg<WriteOnlyTex2D> dual_var_texture{*_blurred_dual_variance};
            
            auto p = thread_xy();
            If (p.x() < _width && p.y() < _height) {
                Auto va = read(color_a_texture, p);
                Auto vb = read(color_b_texture, p);
                Auto diff = va - vb;
                Auto dual_var = 0.25f * diff * diff;
                write(dual_var_texture, p, dual_var);
            };
        });
        
        _blur_dual_variance = std::make_unique<BoxBlur>(device, blur_radius, blur_radius, *_blurred_dual_variance, *_blurred_dual_variance);
        _blur_sample_variance = std::make_unique<BoxBlur>(device, blur_radius, blur_radius, var_sample, *_blurred_sample_variance);
        
        _scale_kernel = device.compile_kernel([&] {
            
            Arg<ReadOnlyTex2D> blurred_dual_var_texture{*_blurred_dual_variance};
            Arg<ReadOnlyTex2D> blurred_sample_var_texture{*_blurred_sample_variance};
            
            auto p = thread_xy();
            auto match_pixel = [&](auto &&in_texture, auto &&out_texture) {
                Auto sv = max(read(blurred_sample_var_texture, p), 1e-6f);
                Auto dv = read(blurred_dual_var_texture, p);
                Auto v = read(in_texture, p);
                write(out_texture, p, min(v * dv / sv, 1e3f));
            };
            
            If (p.x() < _width && p.y() < _height) {
                // Special case handling when "var_sample" and "output" alias...
                if (&var_sample == &output) {  // alias
                    Arg<ReadWriteTex2D> t{var_sample};
                    match_pixel(t, t);
                } else {
                    Arg<ReadOnlyTex2D> in{var_sample};
                    Arg<WriteOnlyTex2D> out{var_sample};
                    match_pixel(in, out);
                }
            };
        });
    }
    
    void operator()(Dispatcher &dispatch) {
        using namespace luisa;
        dispatch(*_dual_variance_kernel, make_uint2(_width, _height));
        dispatch(*_blur_sample_variance);
        dispatch(*_blur_dual_variance);
        dispatch(*_scale_kernel, make_uint2(_width, _height));
    }
};
