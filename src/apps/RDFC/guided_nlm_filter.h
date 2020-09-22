//
// Created by Mike Smith on 2020/8/18.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl_syntax.h>
#include <compute/kernel.h>

#include "box_blur.h"

using luisa::compute::Device;
using luisa::compute::Kernel;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class GuidedNonLocalMeansFilter {

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
    TextureView _accum_texture;
    TextureView _delta_distance_texture;
    TextureView _delta_accum_texture;
    TextureView _delta_blur_temp_texture;
    KernelView _delta_blur_x;
    KernelView _delta_blur_y;
    std::unique_ptr<BoxBlur> _blur;

public:
    // Note: "color" and "output" can be referencing the same texture.
    GuidedNonLocalMeansFilter(
        Device &device, int filter_radius, int patch_radius,
        TextureView color, TextureView var_color, float k_color, float tau,
        TextureView albedo, TextureView var_albedo, TextureView grad_albedo, float k_albedo,
        TextureView normal, TextureView var_normal, TextureView grad_normal, float k_normal,
        TextureView depth, TextureView var_depth, TextureView grad_depth, float k_depth,
        TextureView visibility, TextureView var_vis, TextureView grad_vis, float k_vis,
        TextureView output)
        
        : _width{static_cast<int>(color.width())},
          _height{static_cast<int>(color.height())},
          _filter_radius{filter_radius} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _distance_texture = device.allocate_texture<float>(color.width(), color.height());
        _accum_texture = device.allocate_texture<luisa::float4>(color.width(), color.height());
        
        _distance_kernel = device.compile_kernel("guided_nlm_distance", [&] {
            
            // Note: offset changes from pass to pass, so we bind a pointer to it
            auto d = uniform(&_current_offset);
            auto p = thread_xy();
            If (p.x < _width && p.y < _height) {
                Var target = make_int2(p) + d;
                Var q = make_uint2(
                    select(target.x < 0, -target.x, select(target.x < _width, target.x, 2 * _width - 1 - target.x)),
                    select(target.y < 0, -target.y, select(target.y < _height, target.y, 2 * _height - 1 - target.y)));
                Var var_p = make_float3(var_color.read(p));
                Var var_q = make_float3(var_color.read(q));
                Var var_pq = min(var_p, var_q);
                Var diff = make_float3(color.read(p) - color.read(q));
                constexpr auto epsilon = 1e-4f;
                Var distance = (diff * diff - (var_p + var_pq)) / (epsilon + k_color * k_color * (var_p + var_q));
                Var sum_distance = (distance.r + distance.g + distance.b) * (1.0f / 3.0f);
                _distance_texture.write(q, make_float4(make_float3(sum_distance), 1.0f));
            };
        });
        
        _blur = std::make_unique<BoxBlur>(device, patch_radius, patch_radius, _distance_texture, _distance_texture);
        
        _clear_accum_kernel = device.compile_kernel("guided_nlm_clear_accum", [&] {
            auto p = thread_xy();
            If(p.x < _width && p.y < _height) {
                _accum_texture.write(p, dsl::make_float4(0.0f));
            };
        });
        
        _accum_kernel = device.compile_kernel("guided_nlm_accum", [&] {
            
            // Pointer to offset, will be updated before each launches
            auto d = uniform(&_current_offset);
            Var p = make_int2(thread_xy());
            Var q = p + d;
            If (p.x < _width && p.y < _height && q.x >= 0 && q.x < _width && q.y >= 0 && q.y < _height) {
                
                auto distance_from_feature = [&](TextureView feature, TextureView var_feature, TextureView grad_feature, float k_feature) {
                    Var f_p = make_float3(feature.read(thread_xy()));
                    Var f_q = make_float3(feature.read(make_uint2(q)));
                    Var diff_pq = f_p - f_q;
                    Var var_p = make_float3(var_feature.read(thread_xy()));
                    Var var_q = make_float3(var_feature.read(make_uint2(q)));
                    Var var_pq = min(var_p, var_q);
                    Var grad_p = make_float3(grad_feature.read(thread_xy()));
                    Var d = ((diff_pq * diff_pq) - (var_p + var_pq)) / (k_feature * k_feature * max(tau, max(var_p, grad_p * grad_p)));
                    return max(max(d.x, d.y), d.z);
                };
                
                Var distance_albedo = distance_from_feature(albedo, var_albedo, grad_albedo, k_albedo);
                Var distance_normal = distance_from_feature(normal, var_normal, grad_normal, k_normal);
                Var distance_depth = distance_from_feature(depth, var_depth, grad_depth, k_depth);
                Var distance_vis = distance_from_feature(visibility, var_vis, grad_vis, k_vis);
                
                Var distance_feature = max(max(max(distance_albedo, distance_normal), max(distance_depth, distance_vis)), 0.0f);
                Var w = exp(-distance_feature - (d.x * d.x + d.y * d.y) * 0.075f);
                
                if (!std::isinf(k_color)) {
                    Var distance_color = max(_distance_texture.read(thread_xy()).r, 0.0f);
                    w *= exp(-distance_color);
                }
                
                Var color_q = make_float3(color.read(make_uint2(q)));
                Var accum = _accum_texture.read(thread_xy());
                _accum_texture.write(thread_xy(), make_float4(color_q * w + make_float3(accum), accum.w + w));
            };
        });
        
        _blit_kernel = device.compile_kernel("guided_nlm_blit", [&] {
            auto p = thread_xy();
            If(p.x < _width && p.y < _height) {
                Var accum = _accum_texture.read(p);
                Var filtered = select(accum.w <= 0.0f, dsl::make_float3(0.0f), make_float3(accum) / accum.w);
                output.write(p, make_float4(filtered, 1.0f));
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
