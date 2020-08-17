#include <random>
#include <opencv2/opencv.hpp>
#include <compute/device.h>
#include <compute/dsl.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

void blur(Texture &input, Texture &output, int width, int height, int rx, int ry) noexcept {
    
    Arg<Tex2D<TextureAccess::READ_ONLY>> in{input};
    Arg<Tex2D<TextureAccess::WRITE_ONLY>> out{output};
    
    Auto tx = cast<int>(thread_xy().x());
    Auto ty = cast<int>(thread_xy().y());
    If(tx < width && ty < height) {
        Float4 sum;
        for (auto dx = -rx; dx <= rx; dx++) {
            auto x = tx + dx;
            If(x >= 0 && x < width) {
                sum += read(in, make_uint2(x, ty));
            };
        }
        for (auto dy = -ry; dy <= ry; dy++) {
            auto y = ty + dy;
            If(y >= 0 && y < height) {
                sum += read(in, make_uint2(tx, y));
            };
        }
        write(out, thread_xy(), make_float4(make_float3(sum) / sum.a(), 1.0f));
    };
}

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    auto device = Device::create(&context, "metal");
    
    auto image = cv::imread("data/images/luisa.png", cv::IMREAD_COLOR);
    if (image.type() == CV_8UC3) { cv::cvtColor(image, image, cv::COLOR_BGR2BGRA); }
    
    auto width = static_cast<uint32_t>(image.cols);
    auto height = static_cast<uint32_t>(image.rows);
    
    auto texture = device->allocate_texture<uchar4>(width, height);
    auto temp_texture = device->allocate_texture<uchar4>(width, height);
    
    device->launch([&](Dispatcher &dispatch) noexcept {
        dispatch(texture->copy_from(image.data));
    });
    
    constexpr auto rx = 10;
    constexpr auto ry = 3;
    auto blur_x = device->compile_kernel([&] { blur(*texture, *temp_texture, width, height, rx, 0); });
    auto blur_xx = device->compile_kernel([&] { blur(*temp_texture, *texture, width, height, rx, 0); });
    
    auto blur_y = device->compile_kernel([&] { blur(*texture, *temp_texture, width, height, 0, ry); });
    
    device->launch([&](Dispatcher &dispatch) {
        for (auto i = 0; i < 10; i++) {
            dispatch(*blur_x, make_uint2(width, height));
            dispatch(*blur_xx, make_uint2(width, height));
        }
        dispatch(*blur_y, make_uint2(width, height));
        dispatch(temp_texture->copy_to(image.data));
    });
    device->synchronize();
    
    LUISA_INFO("Saving image...");
    cv::imwrite("data/images/luisa-blur.png", image);
}
