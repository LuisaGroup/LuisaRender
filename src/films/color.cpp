//
// Created by Mike on 2022/1/7.
//

#if defined(LUISA_USE_OPENCV)
#include <opencv2/opencv.hpp>
#endif

#include <luisa-compute.h>
#include <scene/film.h>
#include <scene/pipeline.h>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace luisa::render {

using namespace luisa::compute;

class ColorFilm;

class ColorFilmInstance final : public Film::Instance {

private:
    Image<float> _image;
    Shader2D<Image<float>> _clear_image;

public:
    ColorFilmInstance(Device &device, const ColorFilm *film) noexcept;
    void accumulate(Expr<uint2> pixel, Expr<float3> color) const noexcept override;
    void save(Stream &stream, const std::filesystem::path &path) const noexcept override;
    void clear(CommandBuffer &command_buffer) noexcept override;
};

class ColorFilm final : public Film {

public:
    ColorFilm(Scene *scene, const SceneNodeDesc *desc) noexcept : Film{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<ColorFilmInstance>(pipeline.device(), this);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "color"; }
};

ColorFilmInstance::ColorFilmInstance(Device &device, const ColorFilm *film) noexcept
    : Film::Instance{film},
      _image{device.create_image<float>(PixelStorage::FLOAT4, film->resolution())} {
    Kernel2D clear_image = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    _clear_image = device.compile(clear_image);
}

void ColorFilmInstance::save(Stream &stream, const std::filesystem::path &path) const noexcept {
    auto resolution = node()->resolution();
    auto file_ext = path.extension().string();
    for (auto &c : file_ext) { c = static_cast<char>(tolower(c)); }
#if defined(LUISA_USE_OPENCV)
    cv::Mat image{
        static_cast<int>(resolution.y),
        static_cast<int>(resolution.x),
        CV_32FC4, cv::Scalar::all(0)};
    // TODO: support post-processing pipeline...
    stream << _image.copy_to(image.data) << synchronize();
    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    if (file_ext == ".exr") {// HDR
        cv::imwrite(path.string(), image);
    } else {// LDR
        LUISA_ERROR_WITH_LOCATION(
            "Film extension '{}' is not supported.",
            file_ext);
    }
#else
    std::vector<float> data, rgb;
    data.resize(resolution.y * resolution.x * 4);
    rgb.resize(resolution.y * resolution.x * 3);
    stream << _image.copy_to(data.data()) << synchronize();
    for (int i = 0; i < resolution.x * resolution.y; i++) {
        for (int c = 0; c < 3; c++)
            rgb[i * 3 + c] = data[i * 4 + c];
    }
    if (file_ext == ".exr") {
        const char* err = nullptr;
        int ret = SaveEXR(rgb.data(),
                          resolution.x, resolution.y, 3, 1 /* write as fp16 */, path.string().c_str(), &err);
        if (ret != TINYEXR_SUCCESS) {
            LUISA_ERROR_WITH_LOCATION(
                "Failure when writing image '{}': OpenEXR error: {}", file_ext, err);
        }
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Film extension '{}' is not supported.",
            file_ext);
    }
#endif
}

void ColorFilmInstance::accumulate(Expr<uint2> pixel, Expr<float3> color) const noexcept {
    auto old = _image.read(pixel);
    auto t = old.w + 1.0f;
    _image.write(pixel, make_float4(lerp(old.xyz(), color, 1.0f / t), t));
}

void ColorFilmInstance::clear(CommandBuffer &command_buffer) noexcept {
    command_buffer << _clear_image(_image).dispatch(node()->resolution());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ColorFilm)
