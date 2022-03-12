//
// Created by Mike on 2022/1/7.
//

#include <luisa-compute.h>
#include <base/film.h>
#include <base/pipeline.h>
#include <util/colorspace.h>

namespace luisa::render {

using namespace luisa::compute;

class ColorFilm final : public Film {

private:
    float3 _scale;

public:
    ColorFilm(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Film{scene, desc} {
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
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

class ColorFilmInstance final : public Film::Instance {

private:
    Buffer<float4> _image;
    Buffer<float4> _converted;
    Shader1D<Buffer<float4>> _clear_image;
    Shader1D<Buffer<float4>, Buffer<float4>> _convert_image;

public:
    ColorFilmInstance(Device &device, Pipeline &pipeline, const ColorFilm *film) noexcept;
    void accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept override;
    void clear(CommandBuffer &command_buffer) noexcept override;
    void download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept override;
};

ColorFilmInstance::ColorFilmInstance(Device &device, Pipeline &pipeline, const ColorFilm *film) noexcept
    : Film::Instance{pipeline, film} {
    auto pixel_count = film->resolution().x * film->resolution().y;
    _image = device.create_buffer<float4>(pixel_count);
    _converted = device.create_buffer<float4>(pixel_count);
    Kernel2D clear_image = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };
    _clear_image = device.compile<1>([](BufferFloat4 image) noexcept {
        image.write(dispatch_id().x, make_float4());
    });
    _convert_image = device.compile<1>([this](BufferFloat4 accum, BufferFloat4 out) noexcept {
        auto c = accum.read(dispatch_x());
        auto scale = node<ColorFilm>()->scale();
        out.write(dispatch_x(), make_float4(scale * c.xyz(), 1.f));
    });
}

void ColorFilmInstance::accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept {
    auto width = node()->resolution().x;
    auto valid = !any(isnan(rgb));
    auto pixel_id = pixel.y * width + pixel.x;
    auto old = _image.read(pixel_id);
    auto t = ite(valid, old.w + 1.0f, old.w);
    auto threshold = 65536.0f;
    auto lum = srgb_to_cie_y(rgb);
    auto c = rgb * (threshold / max(lum, threshold));
    auto color = ite(valid, lerp(old.xyz(), c, 1.0f / t), old.xyz());
    _image.write(pixel_id, make_float4(color, t));
}

void ColorFilmInstance::download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept {
    auto pixel_count = node()->resolution().x * node()->resolution().y;
    command_buffer << _convert_image(_image, _converted).dispatch(pixel_count)
                   << _converted.copy_to(framebuffer);
}

void ColorFilmInstance::clear(CommandBuffer &command_buffer) noexcept {
    auto pixel_count = node()->resolution().x * node()->resolution().y;
    command_buffer << _clear_image(_image).dispatch(pixel_count);
}

luisa::unique_ptr<Film::Instance> ColorFilm::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ColorFilmInstance>(pipeline.device(), pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ColorFilm)
