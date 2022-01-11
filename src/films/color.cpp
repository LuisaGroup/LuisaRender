//
// Created by Mike on 2022/1/7.
//

#include <tinyexr.h>

#include <luisa-compute.h>
#include <scene/film.h>
#include <scene/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class ColorFilm;

class ColorFilmInstance final : public Film::Instance {

private:
    Image<float> _image;
    Shader2D<Image<float>> _clear_image;
    float _scale;

public:
    ColorFilmInstance(Device &device, const ColorFilm *film, float scale) noexcept;
    void accumulate(Expr<uint2> pixel, Expr<float3> color) const noexcept override;
    void save(Stream &stream, const std::filesystem::path &path) const noexcept override;
    void clear(CommandBuffer &command_buffer) noexcept override;
};

class ColorFilm final : public Film {

private:
    float _exposure;

public:
    ColorFilm(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Film{scene, desc}, _exposure{desc->property_float_or_default("exposure", 0.0f)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<ColorFilmInstance>(pipeline.device(), this, std::pow(2.0f, _exposure));
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "color"; }
};

ColorFilmInstance::ColorFilmInstance(Device &device, const ColorFilm *film, float scale) noexcept
    : Film::Instance{film},
      _image{device.create_image<float>(PixelStorage::FLOAT4, film->resolution())},
      _scale{scale} {
    Kernel2D clear_image = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    _clear_image = device.compile(clear_image);
}

void ColorFilmInstance::save(Stream &stream, const std::filesystem::path &path) const noexcept {
    auto resolution = node()->resolution();
    auto file_ext = path.extension().string();
    for (auto &c : file_ext) { c = static_cast<char>(tolower(c)); }
    std::vector<float> rgb;
    rgb.resize(resolution.y * resolution.x * 4);
    stream << _image.copy_to(rgb.data()) << synchronize();
    for (auto i = 0; i < resolution.x * resolution.y; i++) {
        for (auto c = 0; c < 3; c++) {
            rgb[i * 3 + c] = _scale * rgb[i * 4 + c];
        }
    }
    if (file_ext == ".exr") {
        const char *err = nullptr;
        auto ret = SaveEXR(
            rgb.data(), static_cast<int>(resolution.x), static_cast<int>(resolution.y),
            3, false, path.string().c_str(), &err);
        if (ret != TINYEXR_SUCCESS) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failure when writing image '{}'. "
                "OpenEXR error: {}", path.string(), err);
        }
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Film extension '{}' is not supported.",
            file_ext);
    }
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
