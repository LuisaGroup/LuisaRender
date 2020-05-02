//
// Created by Mike Smith on 2020/2/2.
//

#include <opencv2/opencv.hpp>
#include <core/film.h>

#include "rgb_film.h"

namespace luisa {

class RGBFilm : public Film {

private:
    std::unique_ptr<Kernel> _postprocess_kernel;

public:
    RGBFilm(Device *device, const ParameterSet &parameters);
    void postprocess(KernelDispatcher &dispatch) override;
    void save(const std::filesystem::path &filename) override;
    
};

LUISA_REGISTER_NODE_CREATOR("RGB", RGBFilm)

void RGBFilm::postprocess(KernelDispatcher &dispatch) {
    auto pixel_count = _resolution.x * _resolution.y;
    dispatch(*_postprocess_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("accumulation_buffer", *_accumulation_buffer);
        encode("pixel_count", pixel_count);
    });
    _accumulation_buffer->synchronize(dispatch);
}

void RGBFilm::save(const std::filesystem::path &filename) {
    cv::Mat image(cv::Size2l{_resolution.x, _resolution.y}, CV_32FC4, _accumulation_buffer->data());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    auto path = std::filesystem::absolute(filename);
    if (path.extension() != ".exr") {
        LUISA_WARNING("File name not ended with .exr: ", filename);
        path += ".exr";
    }
    cv::imwrite(path, image);
}

RGBFilm::RGBFilm(Device *device, const ParameterSet &parameters) : Film{device, parameters} {
    _postprocess_kernel = device->create_kernel("rgb_film_postprocess");
}

}
