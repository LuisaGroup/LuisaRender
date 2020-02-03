//
// Created by Mike Smith on 2020/2/2.
//

#include <opencv2/opencv.hpp>
#include "rgb_film.h"

namespace luisa {

void RGBFilm::postprocess(KernelDispatcher &dispatch) {
    dispatch(*_postprocess_kernel, _resolution, [this](KernelArgumentEncoder &encode) {
        encode("accumulation_buffer", *_accumulation_buffer);
        encode("framebuffer", *_framebuffer);
        encode("resolution", _resolution);
    });
    _framebuffer->synchronize(dispatch);
}

void RGBFilm::save(const std::filesystem::path &filename) {
    cv::Mat image(cv::Size2l{_resolution.x, _resolution.y}, CV_32FC3, _framebuffer->data());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, image);
}

}