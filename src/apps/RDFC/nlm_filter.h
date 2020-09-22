//
// Created by Mike Smith on 2020/8/18.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>
#include <compute/kernel.h>

#include "box_blur.h"

using luisa::compute::Device;
using luisa::compute::KernelView;
using luisa::compute::TextureView;
using luisa::compute::dsl::Function;

class NonLocalMeansFilter {

private:
    int _width;
    int _height;
    int _filter_radius{0};
    luisa::int2 _current_offset{0};
    KernelView _distance_kernel;
    KernelView _clear_accum_kernel;
    KernelView _accum_kernel;
    KernelView _blit_kernel;
    TextureView _distance_texture;
    TextureView _accum_a_texture;
    TextureView _accum_b_texture;
    std::unique_ptr<BoxBlur> _blur;

public:
    // Note: "color" and "output" can be referencing the same texture.
    NonLocalMeansFilter(Device &device, int filter_radius, int patch_radius, float kc,
                        TextureView color, TextureView variance, TextureView color_a, TextureView color_b,
                        TextureView output_a, TextureView output_b)
        : _width{static_cast<int>(color.width())},
          _height{static_cast<int>(color.height())},
          _filter_radius{filter_radius},
          _distance_texture{device.allocate_texture<float>(color.width(), color.height())},
          _accum_a_texture{device.allocate_texture<luisa::float4>(color.width(), color.height())},
          _accum_b_texture{device.allocate_texture<luisa::float4>(color.width(), color.height())} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _blur = std::make_unique<BoxBlur>(device, patch_radius, patch_radius, _distance_texture, _distance_texture);
        
        _distance_kernel = device.compile_kernel("nlm_distance", [&] {
            
            // Note: offset changes from pass to pass, so we bind a pointer to it
            auto d = uniform(&_current_offset);
            
            auto p = thread_xy();
            If (p.x < _width && p.y < _height) {
                Var target = make_int2(p) + d;
                Var q = make_uint2(
                    select(target.x < 0, -target.x, select(target.x < _width, target.x, 2 * _width - 1 - target.x)),
                    select(target.y < 0, -target.y, select(target.y < _height, target.y, 2 * _height - 1 - target.y)));
                Var var_p = make_float3(variance.read(p));
                Var var_q = make_float3(variance.read(q));
                Var var_pq = min(var_p, var_q);
                Var diff = make_float3(color.read(p) - color.read(q));
                constexpr auto epsilon = 1e-6f;
                Var distance = (diff * diff - (var_p + var_pq)) / (epsilon + kc * kc * (var_p + var_q));
                Var sum_distance = (distance.r + distance.g + distance.b) * (1.0f / 3.0f);
                _distance_texture.write(q, make_float4(make_float3(sum_distance), 1.0f));
            };
        });
        
        _clear_accum_kernel = device.compile_kernel("nlm_clear_accum", [&] {
            auto p = thread_xy();
            If(p.x < _width && p.y < _height) {
                _accum_a_texture.write(p, dsl::make_float4(0.0f));
                _accum_b_texture.write(p, dsl::make_float4(0.0f));
            };
        });
        
        _accum_kernel = device.compile_kernel("nlm_accum", [&] {
            
            // Pointer to offset, will be updated before each launches
            auto d = uniform(&_current_offset);
            
            Var p = make_int2(thread_xy());
            Var q = p + d;
            If (p.x < _width && p.y < _height && q.x >= 0 && q.x < _width && q.y >= 0 && q.y < _height) {
                Var weight = exp(-max(_distance_texture.read(thread_xy()).r, 0.0f) - (d.x * d.x + d.y * d.y) * 0.125f);
                auto accumulate = [&](auto &&color_texture, auto &&accum_texture) {
                    Var color_q = make_float3(color_texture.read(make_uint2(q)));
                    Var accum = accum_texture.read(thread_xy());
                    accum_texture.write(thread_xy(), make_float4(color_q * weight + make_float3(accum), accum.w + weight));
                };
                accumulate(color_a, _accum_a_texture);
                accumulate(color_b, _accum_b_texture);
            };
        });
        
        _blit_kernel = device.compile_kernel("nlm_blit", [&] {
            auto p = thread_xy();
            If(p.x < _width && p.y < _height) {
                auto blit = [&](auto &&accum_texture, auto &&output_texture) {
                    Var accum = accum_texture.read(p);
                    Var filtered = select(accum.w <= 0.0f, dsl::make_float3(0.0f), make_float3(accum) / accum.w);
                    output_texture.write(p, make_float4(filtered, 1.0f));
                };
                blit(_accum_a_texture, output_a);
                blit(_accum_b_texture, output_b);
            };
        });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        using namespace luisa;
        dispatch(_clear_accum_kernel.parallelize(make_uint2(_width, _height)));
        for (auto dy = -_filter_radius; dy <= _filter_radius; dy++) {
            for (auto dx = -_filter_radius; dx <= _filter_radius; dx++) {
                _current_offset = make_int2(dx, dy);
                dispatch(_distance_kernel.parallelize(make_uint2(_width, _height)));
                dispatch(*_blur);
                dispatch(_accum_kernel.parallelize(make_uint2(_width, _height)));
            }
        }
        dispatch(_blit_kernel.parallelize(make_uint2(_width, _height)));
    }
};
