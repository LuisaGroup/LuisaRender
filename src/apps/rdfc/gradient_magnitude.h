//
// Created by Mike Smith on 2020/9/8.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>

using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class GradientMagnitude {

private:
    int _width;
    int _height;
    Texture *_output;
    std::shared_ptr<Texture> _result;
    std::shared_ptr<Kernel> _gradient_kernel;

public:
    GradientMagnitude(Device &device, Texture &texture, Texture &output)
        : _width{static_cast<int>(texture.width())},
          _height{static_cast<int>(texture.height())},
          _output{&output} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        if (&texture == &output) { _result = device.allocate_texture<float4>(_width, _height); }
        
        _gradient_kernel = device.compile_kernel("gradient_magnitude", [&] {
            
            Arg<ReadOnlyTex2D> input{texture};
            Arg<WriteOnlyTex2D> grad{&texture == &output ? *_result : output};
            
            Auto p = make_int2(thread_xy());
            
            If (p.x() < _width && p.y() < _height) {
                
                Auto c00 = make_float3(read(input, make_uint2(abs(p.x() - 1), abs(p.y() - 1))));
                Auto c01 = make_float3(read(input, make_uint2(p.x(), abs(p.y() - 1))));
                Auto c02 = make_float3(read(input, make_uint2(select(p.x() == _width - 1, _width - 2, p.x() + 1), abs(p.y() - 1))));
                Auto c10 = make_float3(read(input, make_uint2(abs(p.x() - 1), p.y())));
                Auto c12 = make_float3(read(input, make_uint2(select(p.x() == _width - 1, _width - 2, p.x() + 1), p.y())));
                Auto c20 = make_float3(read(input, make_uint2(abs(p.x() - 1), select(p.y() == _height - 1, _height - 2, p.y() + 1))));
                Auto c21 = make_float3(read(input, make_uint2(p.x(), select(p.y() == _height - 1, _height - 2, p.y() + 1))));
                Auto c22 = make_float3(read(input, make_uint2(select(p.x() == _width - 1, _width - 2, p.x() + 1), select(p.y() == _height - 1, _height - 2, p.y() + 1))));
                
                constexpr auto k0 = 47.0f / 256.0f;
                constexpr auto k1 = 162.0f / 256.0f;
                
                Auto dx = k0 * (c00 - c02) + k1 * (c10 - c12) + k0 * (c20 - c22);
                Auto dy = k0 * (c00 - c20) + k1 * (c01 - c21) + k0 * (c02 - c22);
                Auto g = dsl::sqrt(dx * dx + dy * dy);
                
                write(grad, thread_xy(), make_float4(g, 1.0f));
            };
        });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        dispatch(_gradient_kernel->parallelize(luisa::make_uint2(_width, _height)));
        if (_result != nullptr) { dispatch(_result->copy_to(_output)); }
    }
    
};

