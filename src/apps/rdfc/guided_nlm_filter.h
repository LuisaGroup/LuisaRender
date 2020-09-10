//
// Created by Mike Smith on 2020/8/18.
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

class GuidedNonLocalMeansFilter {

private:
    int _width;
    int _height;
    int _filter_radius{0};
    luisa::int2 _current_offset{0};
    std::shared_ptr<Kernel> _distance_kernel;
    std::shared_ptr<Kernel> _clear_accum_kernel;
    std::shared_ptr<Kernel> _accum_kernel;
    std::shared_ptr<Kernel> _blit_kernel;
    std::shared_ptr<Texture> _distance_texture;
    std::shared_ptr<Texture> _accum_texture;
    std::shared_ptr<Texture> _delta_distance_texture;
    std::shared_ptr<Texture> _delta_accum_texture;
    std::shared_ptr<Texture> _delta_blur_temp_texture;
    std::shared_ptr<Kernel> _delta_blur_x;
    std::shared_ptr<Kernel> _delta_blur_y;
    std::unique_ptr<BoxBlur> _blur;

public:
    // Note: "color" and "output" can be referencing the same texture.
    GuidedNonLocalMeansFilter(
        Device &device, int filter_radius, int patch_radius,
        Texture &color, Texture &var_color, float k_color, float tau,
        Texture &albedo, Texture &var_albedo, Texture &grad_albedo, float k_albedo,
        Texture &normal, Texture &var_normal, Texture &grad_normal, float k_normal,
        Texture &depth, Texture &var_depth, Texture &grad_depth, float k_depth,
        Texture &visibility, Texture &var_vis, Texture &grad_vis, float k_vis,
        Texture &output)
        
        : _width{static_cast<int>(color.width())},
          _height{static_cast<int>(color.height())},
          _filter_radius{filter_radius} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        _distance_texture = device.allocate_texture<float>(color.width(), color.height());
        _accum_texture = device.allocate_texture<luisa::float4>(color.width(), color.height());
        
        _distance_kernel = device.compile_kernel("guided_nlm_distance", [&] {
            
            Arg<ReadOnlyTex2D> variance_texture{var_color};
            Arg<ReadOnlyTex2D> color_texture{color};
            Arg<WriteOnlyTex2D> diff_texture{*_distance_texture};
            
            // Note: offset changes from pass to pass, so we bind a pointer to it
            Arg<int2> d{&_current_offset};
            
            auto p = thread_xy();
            If(p.x() < _width && p.y() < _height) {
                Auto target = make_int2(p) + d;
                Auto q = make_uint2(
                    select(target.x() < 0, -target.x(), select(target.x() < _width, target.x(), 2 * _width - 1 - target.x())),
                    select(target.y() < 0, -target.y(), select(target.y() < _height, target.y(), 2 * _height - 1 - target.y())));
                Auto var_p = make_float3(read(variance_texture, p));
                Auto var_q = make_float3(read(variance_texture, q));
                Auto var_pq = min(var_p, var_q);
                Auto diff = make_float3(read(color_texture, p) - read(color_texture, q));
                constexpr auto epsilon = 1e-4f;
                Auto distance = (diff * diff - (var_p + var_pq)) / (epsilon + k_color * k_color * (var_p + var_q));
                Auto sum_distance = (distance.r() + distance.g() + distance.b()) * (1.0f / 3.0f);
                write(diff_texture, q, make_float4(make_float3(sum_distance), 1.0f));
            };
        });
        
        _blur = std::make_unique<BoxBlur>(device, patch_radius, patch_radius, *_distance_texture, *_distance_texture);
        
        _clear_accum_kernel = device.compile_kernel("guided_nlm_clear_accum", [&] {
            Arg<WriteOnlyTex2D> accum_texture{*_accum_texture};
//            Arg<WriteOnlyTex2D> delta_accum_texture{*_delta_accum_texture};
            auto p = thread_xy();
            If(p.x() < _width && p.y() < _height) {
                write(accum_texture, p, dsl::make_float4(0.0f));
//                write(delta_accum_texture, p, dsl::make_float4(0.0f));
            };
        });
        
        _accum_kernel = device.compile_kernel("guided_nlm_accum", [&] {
            
            Arg<ReadOnlyTex2D> color_texture{color};
            Arg<ReadWriteTex2D> accum_texture{*_accum_texture};
            
            // Pointer to offset, will be updated before each launches
            Arg<int2> d{&_current_offset};
            
            Auto p = make_int2(thread_xy());
            Auto q = p + d;
            If(p.x() < _width && p.y() < _height && q.x() >= 0 && q.x() < _width && q.y() >= 0 && q.y() < _height) {
                
                auto distance_from_feature = [&](Texture &feature, Texture &var_feature, Texture &grad_feature, float k_feature) {
                    Arg<ReadOnlyTex2D> feature_texture{feature};
                    Arg<ReadOnlyTex2D> var_feature_texture{var_feature};
                    Arg<ReadOnlyTex2D> grad_feature_texture{grad_feature};
                    Auto f_p = make_float3(read(feature_texture, thread_xy()));
                    Auto f_q = make_float3(read(feature_texture, make_uint2(q)));
                    Auto diff_pq = f_p - f_q;
                    Auto var_p = make_float3(read(var_feature_texture, thread_xy()));
                    Auto var_q = make_float3(read(var_feature_texture, make_uint2(q)));
                    Auto var_pq = min(var_p, var_q);
                    Auto grad_p = make_float3(read(grad_feature_texture, thread_xy()));
                    Auto d = ((diff_pq * diff_pq) - (var_p + var_pq)) / (k_feature * k_feature * max(tau, max(var_p, grad_p * grad_p)));
                    return max(max(d.x(), d.y()), d.z());
                };
                
                Auto distance_albedo = distance_from_feature(albedo, var_albedo, grad_albedo, k_albedo);
                Auto distance_normal = distance_from_feature(normal, var_normal, grad_normal, k_normal);
                Auto distance_depth = distance_from_feature(depth, var_depth, grad_depth, k_depth);
                Auto distance_vis = distance_from_feature(visibility, var_vis, grad_vis, k_vis);
                
                Auto distance_feature = max(max(max(distance_albedo, distance_normal), max(distance_depth, distance_vis)), 0.0f);
                Auto w = exp(-distance_feature);
                
                if (!std::isinf(k_color)) {
                    Arg<ReadOnlyTex2D> blurred_distance_texture{*_distance_texture};
                    Auto distance_color = max(read(blurred_distance_texture, thread_xy()).r(), 0.0f);
                    w *= exp(-distance_color);
                }
                
                Auto color_q = make_float3(read(color_texture, make_uint2(q)));
                Auto accum = read(accum_texture, thread_xy());
                write(accum_texture, thread_xy(), make_float4(color_q * w + make_float3(accum), accum.w() + w));
            };
        });
        
        _blit_kernel = device.compile_kernel("guided_nlm_blit", [&] {
            Arg<ReadOnlyTex2D> accum_texture{*_accum_texture};
            Arg<WriteOnlyTex2D> output_texture{output};
            
            auto p = thread_xy();
            If(p.x() < _width && p.y() < _height) {
                Auto accum = read(accum_texture, p);
                Auto filtered = select(accum.w() <= 0.0f, dsl::make_float3(0.0f), make_float3(accum) / accum.w());
                write(output_texture, p, make_float4(filtered, 1.0f));
            };
        });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        using namespace luisa;
        dispatch(_clear_accum_kernel->parallelize(make_uint2(_width, _height)));
        for (auto dy = -_filter_radius; dy <= _filter_radius; dy++) {
            for (auto dx = -_filter_radius; dx <= _filter_radius; dx++) {
                _current_offset = make_int2(dx, dy);
                dispatch(_distance_kernel->parallelize(make_uint2(_width, _height)));
                dispatch(*_blur);
                dispatch(_accum_kernel->parallelize(make_uint2(_width, _height)));
            }
        }
        dispatch(_blit_kernel->parallelize(make_uint2(_width, _height)));
    }
};
