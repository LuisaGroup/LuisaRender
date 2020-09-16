//
// Created by Mike Smith on 2020/9/8.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>

using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::KernelView;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class GradientMagnitude {

private:
    int _width;
    int _height;
    TextureView _output;
    TextureView _result;
    KernelView _gradient_kernel;

public:
    GradientMagnitude(Device &device, TextureView texture, TextureView output)
        : _width{static_cast<int>(texture.width())},
          _height{static_cast<int>(texture.height())},
          _output{std::move(output)} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        auto alias = texture.texture() == _output.texture();
        if (alias) { _result = device.allocate_texture<float4>(_width, _height); }
        
        _gradient_kernel = device.compile_kernel("gradient_magnitude", [&] {
            
            Var p = make_int2(thread_xy());
            If (p.x < _width && p.y < _height) {
                
                Var c00 = make_float3(texture.read(make_uint2(abs(p.x - 1), abs(p.y - 1))));
                Var c01 = make_float3(texture.read(make_uint2(p.x, abs(p.y - 1))));
                Var c02 = make_float3(texture.read(make_uint2(select(p.x == _width - 1, _width - 2, p.x + 1), abs(p.y - 1))));
                Var c10 = make_float3(texture.read(make_uint2(abs(p.x - 1), p.y)));
                Var c12 = make_float3(texture.read(make_uint2(select(p.x == _width - 1, _width - 2, p.x + 1), p.y)));
                Var c20 = make_float3(texture.read(make_uint2(abs(p.x - 1), select(p.y == _height - 1, _height - 2, p.y + 1))));
                Var c21 = make_float3(texture.read(make_uint2(p.x, select(p.y == _height - 1, _height - 2, p.y + 1))));
                Var c22 = make_float3(texture.read(make_uint2(select(p.x == _width - 1, _width - 2, p.x + 1), select(p.y == _height - 1, _height - 2, p.y + 1))));
                
                constexpr auto k0 = 47.0f / 256.0f;
                constexpr auto k1 = 162.0f / 256.0f;
                
                Var dx = k0 * (c00 - c02) + k1 * (c10 - c12) + k0 * (c20 - c22);
                Var dy = k0 * (c00 - c20) + k1 * (c01 - c21) + k0 * (c02 - c22);
                Var g = dsl::sqrt(dx * dx + dy * dy);
                
                if (alias) {
                    _result.write(thread_xy(), make_float4(g, 1.0f));
                } else {
                    _output.write(thread_xy(), make_float4(g, 1.0f));
                }
            };
        });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        dispatch(_gradient_kernel.parallelize(luisa::make_uint2(_width, _height)));
        if (!_result.empty() && !_output.empty()) { dispatch(_result.copy_to(_output)); }
    }
};
