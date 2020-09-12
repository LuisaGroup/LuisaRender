//
// Created by Mike Smith on 2020/8/17.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>

using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::TextureView;
using luisa::compute::dsl::Function;
using luisa::compute::dsl::Var;

class BoxBlur {

private:
    int _width;
    int _height;
    KernelView _blur_x;
    KernelView _blur_y;
    TextureView _temp;

public:
    // Note: "input" and "output" may alias
    BoxBlur(Device &device, int rx, int ry, TextureView input, TextureView output) noexcept
        : _width{static_cast<int>(input.width())},
          _height{static_cast<int>(input.height())},
          _temp{device.allocate_texture<luisa::float4>(input.width(), input.height())} {
    
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        auto box_blur_x_or_y = [](int rx, int ry, TextureView in, TextureView out) noexcept {
            
            auto width = static_cast<int>(in.width());
            auto height = static_cast<int>(in.height());
            Var p = make_int2(thread_xy());
            If (all(p < dsl::make_int2(width, height))) {
                Var sum = dsl::make_float3(0.0f);
                for (auto dy = -ry; dy <= ry; dy++) {
                    for (auto dx = -rx; dx <= rx; dx++) {
                        Var<int> x = p.x() + Expr{dx};
                        Var<int> y = p.y() + Expr{dy};
                        if (rx != 0) { x = Expr{select(x < 0, -x, select(x < width, x, 2 * width - 1 - x))}; }
                        if (ry != 0) { y = Expr{select(y < 0, -y, select(y < height, y, 2 * height - 1 - y))}; }
                        sum += make_float3(in.read(make_uint2(x, y)));
                    }
                }
                auto support = (2 * rx + 1) * (2 * ry + 1);
                out.write(thread_xy(), make_float4(sum * (1.0f / static_cast<float>(support)), 1.0f));
            };
        };
        
        _blur_x = device.compile_kernel("box_blur_x", [&] { box_blur_x_or_y(rx, 0, input, _temp); });
        _blur_y = device.compile_kernel("box_blur_y", [&] { box_blur_x_or_y(0, ry, _temp, output); });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        using namespace luisa;
        dispatch(_blur_x.parallelize(make_uint2(_width, _height)));
        dispatch(_blur_y.parallelize(make_uint2(_width, _height)));
    }
};
