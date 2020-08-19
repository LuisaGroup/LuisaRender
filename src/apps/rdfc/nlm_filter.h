//
// Created by Mike Smith on 2020/8/18.
//

#pragma once

#include <compute/dsl.h>
#include <compute/kernel.h>
#include <compute/device.h>

#include "box_blur.h"

using luisa::compute::Kernel;
using luisa::compute::Device;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class NonLocalMeansFilter {

private:
    Device *_device;
    int _width;
    int _height;
    int _filter_radius{0};
    luisa::int2 _current_offset{0};
    std::unique_ptr<Kernel> _distance_kernel;
    std::unique_ptr<Kernel> _clear_accum_kernel;
    std::unique_ptr<Kernel> _accum_kernel;
    std::unique_ptr<Kernel> _blit_kernel;
    std::unique_ptr<Texture> _distance_texture;
    std::unique_ptr<Texture> _accum_a_texture;
    std::unique_ptr<Texture> _accum_b_texture;
    std::unique_ptr<BoxBlur> _blur;

public:
    // Note: "color" and "output" can be referencing the same texture.
    NonLocalMeansFilter(Device &device, int filter_radius, int patch_radius, float kc,
                        Texture &color, Texture &variance, Texture &color_a, Texture &color_b,
                        Texture &output_a, Texture &output_b)
        : _device{&device},
          _width{static_cast<int>(color.width())},
          _height{static_cast<int>(color.height())},
          _filter_radius{filter_radius},
          _distance_texture{device.allocate_texture<float>(color.width(), color.height())},
          _accum_a_texture{device.allocate_texture<luisa::float4>(color.width(), color.height())},
          _accum_b_texture{device.allocate_texture<luisa::float4>(color.width(), color.height())} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _blur = std::make_unique<BoxBlur>(device, patch_radius, patch_radius, *_distance_texture, *_distance_texture);
        
        _distance_kernel = device.compile_kernel([&] {
            
            Arg<ReadOnlyTex2D> variance_texture{variance};
            Arg<ReadOnlyTex2D> color_texture{color};
            Arg<WriteOnlyTex2D> diff_texture{*_distance_texture};
            
            // Note: offset changes from pass to pass, so we bind a pointer to it
            Arg<int2> d{&_current_offset};
            
            auto p = thread_xy();
            If (p.x() < _width && p.y() < _height) {
                Auto target = make_int2(p) + d;
                Auto q = make_uint2(
                    select(target.x() < 0, -target.x(), select(target.x() < _width, target.x(), 2 * _width - 1 - target.x())),
                    select(target.y() < 0, -target.y(), select(target.y() < _height, target.y(), 2 * _height - 1 - target.y())));
                Auto var_p = make_float3(read(variance_texture, p));
                Auto var_q = make_float3(read(variance_texture, q));
                Auto var_pq = min(var_p, var_q);
                Auto diff = make_float3(read(color_texture, p) - read(color_texture, q));
                constexpr auto epsilon = 1e-4f;
                Auto distance = (diff * diff - (var_p + var_pq)) / (epsilon + kc * kc * (var_p + var_q));
                Auto sum_distance = (distance.r() + distance.g() + distance.b()) * (1.0f / 3.0f);
                write(diff_texture, q, make_float4(make_float3(sum_distance), 1.0f));
            };
        });
        
        _clear_accum_kernel = device.compile_kernel([&] {
            Arg<WriteOnlyTex2D> accum_a_texture{*_accum_a_texture};
            Arg<WriteOnlyTex2D> accum_b_texture{*_accum_b_texture};
            auto p = thread_xy();
            If (p.x() < _width && p.y() < _height) {
                write(accum_a_texture, p, dsl::make_float4(0.0f));
                write(accum_b_texture, p, dsl::make_float4(0.0f));
            };
        });
        
        _accum_kernel = device.compile_kernel([&] {
            
            Arg<ReadOnlyTex2D> blurred_distance_texture{*_distance_texture};
            Arg<ReadOnlyTex2D> color_a_texture{color_a};
            Arg<ReadOnlyTex2D> color_b_texture{color_b};
            Arg<ReadWriteTex2D> accum_a_texture{*_accum_a_texture};
            Arg<ReadWriteTex2D> accum_b_texture{*_accum_b_texture};
            
            // Pointer to offset, will be updated before each launches
            Arg<int2> d{&_current_offset};
            
            Auto p = make_int2(thread_xy());
            Auto q = p + d;
            If (p.x() < _width && p.y() < _height && q.x() >= 0 && q.x() < _width && q.y() >= 0 && q.y() < _height) {
                
                auto support = static_cast<float>(2 * patch_radius - 1);
                Auto weight = exp(-max(read(blurred_distance_texture, thread_xy()).r() * (1.0f / (support * support)), 0.0f));
                
                auto accumulate = [&q, &weight](auto &&color_texture, auto &&accum_texture) {
                    Auto color_q = make_float3(read(color_texture, make_uint2(q)));
                    Auto accum = read(accum_texture, thread_xy());
                    write(accum_texture, thread_xy(), make_float4(color_q * weight + make_float3(accum), accum.w() + weight));
                };
                accumulate(color_a_texture, accum_a_texture);
                accumulate(color_b_texture, accum_b_texture);
            };
        });
        
        _blit_kernel = device.compile_kernel([&] {
            
            Arg<ReadOnlyTex2D> accum_a_texture{*_accum_a_texture};
            Arg<ReadOnlyTex2D> accum_b_texture{*_accum_b_texture};
            Arg<WriteOnlyTex2D> output_a_texture{output_a};
            Arg<WriteOnlyTex2D> output_b_texture{output_b};
            
            auto p = thread_xy();
            If (p.x() < _width && p.y() < _height) {
                auto blit = [&p](auto &&accum_texture, auto &&output_texture) {
                    Auto accum = read(accum_texture, p);
                    Auto filtered = select(accum.w() <= 0.0f, dsl::make_float3(0.0f), make_float3(accum) / accum.w());
                    write(output_texture, p, make_float4(filtered, 1.0f));
                };
                blit(accum_a_texture, output_a_texture);
                blit(accum_b_texture, output_b_texture);
            };
        });
    }
    
    void apply() {
        using namespace luisa;
        _device->launch(*_clear_accum_kernel, make_uint2(_width, _height));
        for (auto dy = -_filter_radius; dy <= _filter_radius; dy++) {
            _device->launch([this, dy](Dispatcher &dispatch) {
                for (auto dx = -_filter_radius; dx <= _filter_radius; dx++) {
                    _current_offset = make_int2(dx, dy);
                    dispatch(*_distance_kernel, make_uint2(_width, _height));
                    dispatch(*_blur);
                    dispatch(*_accum_kernel, make_uint2(_width, _height));
                }
            });
        }
        _device->launch(*_blit_kernel, make_uint2(_width, _height));
    }
};
