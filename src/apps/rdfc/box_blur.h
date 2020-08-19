//
// Created by Mike Smith on 2020/8/17.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>

using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class BoxBlur {

private:
    int _width;
    int _height;
    std::unique_ptr<Kernel> _blur_x;
    std::unique_ptr<Kernel> _blur_y;
    std::unique_ptr<Texture> _temp;

public:
    // Note: "input" and "output" may alias
    BoxBlur(Device &device, int rx, int ry, Texture &input, Texture &output) noexcept
        : _width{static_cast<int>(input.width())},
          _height{static_cast<int>(input.height())},
          _temp{device.allocate_texture<luisa::float4>(input.width(), input.height())} {
        
        auto box_blur_x_or_y = [](int rx, int ry, Texture &input, Texture &output) noexcept {
            
            using namespace luisa;
            using namespace luisa::compute;
            using namespace luisa::compute::dsl;
            
            Arg<ReadOnlyTex2D> in{input};
            Arg<WriteOnlyTex2D> out{output};
            
            auto width = static_cast<int>(input.width());
            auto height = static_cast<int>(input.height());
            Auto p = make_int2(thread_xy());
            If (p.x() < width && p.y() < height) {
                Float3 sum{0.0f};
                for (auto dy = -ry; dy <= ry; dy++) {
                    for (auto dx = -rx; dx <= rx; dx++) {
                        Auto x = p.x() + dx;
                        Auto y = p.y() + dy;
                        if (rx != 0) { x = select(x < 0, -x, select(x < width, x, 2 * width - 1 - x)); }
                        if (ry != 0) { y = select(y < 0, -y, select(y < height, y, 2 * height - 1 - y)); }
                        sum += make_float3(read(in, make_uint2(x, y)));
                    }
                }
                auto support = (2 * rx + 1) * (2 * ry + 1);
                write(out, thread_xy(), make_float4(sum * (1.0f / static_cast<float>(support)), 1.0f));
            };
        };
        
        _blur_x = device.compile_kernel([&] { box_blur_x_or_y(rx, 0, input, *_temp); });
        _blur_y = device.compile_kernel([&] { box_blur_x_or_y(0, ry, *_temp, output); });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        using namespace luisa;
        dispatch(*_blur_x, make_uint2(_width, _height));
        dispatch(*_blur_y, make_uint2(_width, _height));
    }
};
