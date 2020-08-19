//
// Created by Mike Smith on 2020/8/17.
//

#include <opencv2/opencv.hpp>

#include "dual_buffer_variance.h"
#include "nlm_filter.h"
#include "gaussian_blur.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    auto device = Device::create(&context, "metal");
    
    {  // playground
        auto image = cv::imread("data/images/luisa.png", cv::IMREAD_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
        auto texture = device->allocate_texture<uchar4>(image.cols, image.rows);
        device->launch(texture->copy_from(image.data));
        GaussianBlur blur{*device, 2.0f, 25.0f, *texture, *texture};
        device->launch(blur);
        device->launch(texture->copy_to(image.data));
        device->synchronize();
        cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
        cv::imwrite("data/images/luisa-gaussian-blur.png", image);
    }
    
    auto feature_name = "albedo";
    
    auto load_image = [](const std::string &path) {
        LUISA_INFO("Loading image: \"", path, "\"...");
        auto image = cv::imread(path, cv::IMREAD_UNCHANGED);
        cv::cvtColor(image, image, image.channels() == 1 ? cv::COLOR_GRAY2RGBA : cv::COLOR_BGR2RGBA);
        const auto size = image.rows * image.cols * image.channels();
        for (auto i = 0; i < size; i++) {
            auto &&v = reinterpret_cast<float *>(image.data)[i];
            if (std::isnan(v)) { v = 0.0f; }
            if (std::isinf(v)) { v = 1e6f; }
        }
        return image;
    };
    
    auto color_image = load_image(serialize("data/images/", feature_name, ".exr"));
    auto color_a_image = load_image(serialize("data/images/", feature_name, "A.exr"));
    auto color_b_image = load_image(serialize("data/images/", feature_name, "B.exr"));
    auto variance_image = load_image(serialize("data/images/", feature_name, "Variance.exr"));
    
    auto width = static_cast<uint32_t>(color_image.cols);
    auto height = static_cast<uint32_t>(color_image.rows);
    auto color_texture = device->allocate_texture<float4>(width, height);
    auto color_a_texture = device->allocate_texture<float4>(width, height);
    auto color_b_texture = device->allocate_texture<float4>(width, height);
    auto variance_texture = device->allocate_texture<float4>(width, height);
    
    device->launch([&](Dispatcher &dispatch) {
        dispatch(color_texture->copy_from(color_image.data));
        dispatch(color_a_texture->copy_from(color_a_image.data));
        dispatch(color_b_texture->copy_from(color_b_image.data));
        dispatch(variance_texture->copy_from(variance_image.data));
    });
    
    DualBufferVariance dual_variance{*device, 10, *variance_texture, *color_a_texture, *color_b_texture, *variance_texture};
    NonLocalMeansFilter filter{*device, 5, 3, 1.0f, *color_texture, *variance_texture, *color_a_texture, *color_b_texture, *color_a_texture, *color_b_texture};
    
    auto add_half_buffers = device->compile_kernel([&] {
        Arg<ReadOnlyTex2D> color_a{*color_a_texture};
        Arg<ReadOnlyTex2D> color_b{*color_b_texture};
        Arg<WriteOnlyTex2D> color{*color_texture};
        auto p = thread_xy();
        If (p.x() < width && p.y() < height) {
            write(color, p, make_float4(0.5f * make_float3(read(color_a, p) + read(color_b, p)), 1.0f));
        };
    });
    
    device->launch(dual_variance);
    filter.apply();
    device->launch([&](Dispatcher &dispatch) {
        dispatch(variance_texture->copy_to(variance_image.data));
        dispatch(*add_half_buffers, make_uint2(width, height));
        dispatch(color_texture->copy_to(color_image.data));
    });
    
    device->synchronize();
    LUISA_INFO("Done.");
    
    cv::cvtColor(variance_image, variance_image, cv::COLOR_RGBA2BGR);
    cv::cvtColor(color_image, color_image, cv::COLOR_RGBA2BGR);
    cv::imwrite(serialize("data/images/", feature_name, "-nlm.exr"), color_image);
    cv::imwrite(serialize("data/images/", feature_name, "Variance-dual.exr"), variance_image);
}
