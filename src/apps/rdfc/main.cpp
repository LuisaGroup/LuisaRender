//
// Created by Mike Smith on 2020/8/17.
//

#include <opencv2/opencv.hpp>
#include "box_blur.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    auto device = Device::create(&context, "metal");
    
    auto cv_image = cv::imread("data/images/luisa.png", cv::IMREAD_COLOR);
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGBA);
    
    auto width = static_cast<uint32_t>(cv_image.cols);
    auto height = static_cast<uint32_t>(cv_image.rows);
    auto texture = device->allocate_texture<uchar4>(width, height);
    auto temp_texture = device->allocate_texture<uchar4>(width, height);
    BoxBlur blur{*device, 20, 10, *texture, *temp_texture};
    
    device->launch([&](Dispatcher &dispatch) noexcept {
        dispatch(texture->copy_from(cv_image.data));
        dispatch(blur);
        dispatch(texture->copy_to(cv_image.data));
    });
    
    device->synchronize();
    
    cv::cvtColor(cv_image, cv_image, cv::COLOR_RGBA2BGR);
    cv::imwrite("data/images/luisa-box-blur.png", cv_image);
}
