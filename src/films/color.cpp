//
// Created by Mike on 2022/1/7.
//

#include <tinyexr.h>

#include <luisa-compute.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class ColorFilm final : public Film {

private:
    std::array<float, 3> _scale{};
    bool _fp16{};

public:
    ColorFilm(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Film{scene, desc},
          _fp16{desc->property_bool_or_default("fp16", false)} {
        auto exposure = desc->property_float3_or_default(
            "exposure", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "exposure", 0.0f));
            }));
        _scale[0] = std::pow(2.0f, exposure.x);
        _scale[1] = std::pow(2.0f, exposure.y);
        _scale[2] = std::pow(2.0f, exposure.z);
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto fp16() const noexcept { return _fp16; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "color"; }
};

class ColorFilmInstance final : public Film::Instance {

private:
    Image<float> _image;
    Shader2D<Image<float>> _clear_image;

public:
    ColorFilmInstance(Device &device, Pipeline &pipeline, const ColorFilm *film) noexcept;
    void accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept override;
    void save(Stream &stream, const std::filesystem::path &path) const noexcept override;
    void clear(CommandBuffer &command_buffer) noexcept override;
};

ColorFilmInstance::ColorFilmInstance(Device &device, Pipeline &pipeline, const ColorFilm *film) noexcept
    : Film::Instance{pipeline, film},
      _image{device.create_image<float>(
          PixelStorage::FLOAT4, film->resolution())} {
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
    rgb.resize(resolution.x * resolution.y * 4);
    stream << _image.copy_to(rgb.data()) << synchronize();
    auto film = static_cast<const ColorFilm *>(node());
    auto scale = film->scale();
    for (auto i = 0; i < resolution.x * resolution.y; i++) {
        for (auto c = 0; c < 3; c++) {
            rgb[i * 3 + c] = scale[c] * rgb[i * 4 + c];
        }
    }
    if (file_ext == ".exr") {
        const char *err = nullptr;
        auto ret = SaveEXR(
            rgb.data(), static_cast<int>(resolution.x), static_cast<int>(resolution.y),
            3, film->fp16(), path.string().c_str(), &err);
        if (ret != TINYEXR_SUCCESS) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failure when writing image '{}'. "
                "OpenEXR error: {}",
                path.string(), err);
        }
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Film extension '{}' is not supported.",
            file_ext);
    }
}

void ColorFilmInstance::accumulate(Expr<uint2> pixel, Expr<float3> rgb_in) const noexcept {
    auto rgb = ite(any(isnan(rgb_in)), 0.0f, rgb_in);
    auto old = _image.read(pixel);
    auto t = old.w + 1.0f;
    _image.write(pixel, make_float4(lerp(old.xyz(), rgb, 1.0f / t), t));
}

void ColorFilmInstance::clear(CommandBuffer &command_buffer) noexcept {
    command_buffer << _clear_image(_image).dispatch(node()->resolution());
}

luisa::unique_ptr<Film::Instance> ColorFilm::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ColorFilmInstance>(pipeline.device(), pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ColorFilm)
