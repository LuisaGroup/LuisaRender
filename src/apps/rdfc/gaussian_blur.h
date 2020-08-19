//
// Created by Mike Smith on 2020/8/19.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>

using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class GaussianBlur {

private:
    int _width;
    int _height;
    std::unique_ptr<Texture> _temp;
    std::unique_ptr<Kernel> _blur_x_kernel;
    std::unique_ptr<Kernel> _blur_y_kernel;

public:
    GaussianBlur(Device &device, float sigma_x, float sigma_y, Texture &input, Texture &output)
        : _width{static_cast<int>(input.width())},
          _height{static_cast<int>(input.height())},
          _temp{device.allocate_texture<luisa::float4>(input.width(), input.height())} {
        
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        
        auto find_radius = [](float sigma, int size) {
            for (auto d = 0; d < size; d++) { if (std::exp(-static_cast<float>(d * d) / (2.0f * sigma * sigma)) < 0.05f) { return d; }}
            return size;
        };
        
        auto width = static_cast<int>(input.width());
        auto height = static_cast<int>(input.height());
        auto rx = find_radius(sigma_x, width);
        auto ry = find_radius(sigma_y, height);
        LUISA_INFO("Gaussian filter sigma = (", sigma_x, ", ", sigma_y, "), radius = (", rx, ", ", ry, ").");
        
        auto blur_x_or_y = [width, height](int rx, int ry, float sigma_x, float sigma_y, Texture &input, Texture &output) noexcept {
            
            Arg<ReadOnlyTex2D> in{input};
            Arg<WriteOnlyTex2D> out{output};
            Auto p = make_int2(thread_xy());
            If (p.x() < width && p.y() < height) {
                Auto sum = dsl::make_float4(0.0f);
                for (auto dy = -ry; dy <= ry; dy++) {
                    for (auto dx = -rx; dx <= rx; dx++) {
                        Auto x = p.x() + dx;
                        Auto y = p.y() + dy;
                        if (rx != 0) { x = select(x < 0, -x, select(x < width, x, 2 * width - 1 - x)); }
                        if (ry != 0) { y = select(y < 0, -y, select(y < height, y, 2 * height - 1 - y)); }
                        auto weight = std::exp(-static_cast<float>(dx * dx + dy * dy) / (2.0f * (sigma_x * sigma_x + sigma_y * sigma_y)));
                        sum += make_float4(weight * make_float3(read(in, make_uint2(x, y))), weight);
                    }
                }
                write(out, thread_xy(), make_float4(make_float3(sum) / sum.w(), 1.0f));
            };
        };
        
        _blur_x_kernel = device.compile_kernel([&] { blur_x_or_y(rx, 0, sigma_x, 0.0f, input, *_temp); });
        _blur_y_kernel = device.compile_kernel([&] { blur_x_or_y(0, ry, 0.0f, sigma_y, *_temp, output); });
    }
    
    void operator()(Dispatcher &dispatch) noexcept {
        using namespace luisa;
        dispatch(*_blur_x_kernel, make_uint2(_width, _height));
        dispatch(*_blur_y_kernel, make_uint2(_width, _height));
    }
    
};
