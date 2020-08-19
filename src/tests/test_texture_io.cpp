#include <random>
#include <opencv2/opencv.hpp>
#include <compute/device.h>
#include <compute/dsl.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    auto device = Device::create(&context, "metal");
    
    constexpr auto width = 1280u;
    constexpr auto height = 720u;
    
    auto texture = device->allocate_texture<uchar4>(width, height);
    auto buffer = device->allocate_buffer<uchar4>(width * height);
    
    auto linear_to_srgb = [&](Float3 linear) {
        auto gamma = [](auto u) { return select(u <= 0.0031308f, 12.92f * u, 1.055f * pow(u, 1.0f / 2.4f) - 0.055f); };
        return make_float3(gamma(linear.r()), gamma(linear.g()), gamma(linear.b()));
    };
    
    auto kernel = device->compile_kernel([&] {
        Arg<WriteOnlyTex2D> image{*texture};
        Arg<uint2> image_size{make_uint2(width, height)};
        Auto txy = thread_xy();
        If(txy.x() < image_size.x() && txy.y() < image_size.y()) {
            Auto xy_f = make_float2(txy);
            Auto size_f = make_float2(image_size) - 1.0f;
            Auto color = make_float4(linear_to_srgb(make_float3(xy_f / size_f, 1.0f)), 1.0f);
            write(image, txy, color);
        };
    });
    
    cv::Mat image;
    image.create(cv::Size{width, height}, CV_8UC4);
    
    device->launch([&](Dispatcher &d) {
        d(*kernel, make_uint2(width, height));
        d(texture->copy_to(buffer));
        d(buffer.copy_to(image.data));
    });
    device->synchronize();
    
    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    cv::imwrite("data/images/test.png", image);
}
