//
// Created by Mike Smith on 2020/8/17.
//

#include <opencv2/opencv.hpp>
#include "nlm_filter.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    auto device = Device::create(&context, "metal");
    
    auto feature_name = "visibility";
    
    auto color_image = cv::imread(serialize("data/images/", feature_name, ".exr"), cv::IMREAD_UNCHANGED);
    auto variance_image = cv::imread(serialize("data/images/", feature_name, "Variance.exr"), cv::IMREAD_UNCHANGED);
    
    cv::cvtColor(color_image, color_image, color_image.channels() == 1 ? cv::COLOR_GRAY2RGBA : cv::COLOR_BGR2RGBA);
    cv::cvtColor(variance_image, variance_image, variance_image.channels() == 1 ? cv::COLOR_GRAY2RGBA : cv::COLOR_BGR2RGBA);
    
    auto width = static_cast<uint32_t>(color_image.cols);
    auto height = static_cast<uint32_t>(color_image.rows);
    auto color_texture = device->allocate_texture<float4>(width, height);
    auto variance_texture = device->allocate_texture<float4>(width, height);
    
    NonLocalMeansFilter filter{*device, 10, 3, 1.0f, *color_texture, *variance_texture};
    
    device->launch([&](Dispatcher &dispatch) noexcept {
        dispatch(color_texture->copy_from(color_image.data));
        dispatch(variance_texture->copy_from(variance_image.data));
        dispatch(filter);
        dispatch(color_texture->copy_to(color_image.data));
    });
    
    device->synchronize();
    
    cv::cvtColor(color_image, color_image, cv::COLOR_RGBA2BGR);
    cv::imwrite(serialize("data/images/", feature_name, "-nlm.exr"), color_image);
}
