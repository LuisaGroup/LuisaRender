//
// Created by Mike on 2022/1/7.
//

#include <luisa-compute.h>
#include <base/film.h>
#include <base/pipeline.h>
#include <util/atomic.h>
#include <util/colorspace.h>

namespace luisa::render {

using namespace luisa::compute;

class AtomicColorFilm final : public Film {

private:
    float3 _scale;

public:
    AtomicColorFilm(Scene *scene, const SceneNodeDesc *desc) noexcept
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

class AtomicColorFilmInstance final : public Film::Instance {

private:
    Buffer<uint> _image;
    Buffer<float4> _converted;
    Shader1D<Buffer<uint>> _clear_image;
    Shader1D<Buffer<uint>, Buffer<float4>> _convert_image;

public:
    AtomicColorFilmInstance(Device &device, Pipeline &pipeline, const AtomicColorFilm *film) noexcept;
    void accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept override;
    void clear(CommandBuffer &command_buffer) noexcept override;
    void download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept override;
};

AtomicColorFilmInstance::AtomicColorFilmInstance(Device &device, Pipeline &pipeline, const AtomicColorFilm *film) noexcept
    : Film::Instance{pipeline, film} {
    auto resolution = node()->resolution();
    auto pixel_count = resolution.x * resolution.y;
    _image = pipeline.device().create_buffer<uint>(pixel_count * 4u);
    _converted = pipeline.device().create_buffer<float4>(pixel_count);
    _clear_image = device.compile<1>([](BufferUInt image) noexcept {
        image.write(dispatch_x() * 4u + 0u, 0u);
        image.write(dispatch_x() * 4u + 1u, 0u);
        image.write(dispatch_x() * 4u + 2u, 0u);
        image.write(dispatch_x() * 4u + 3u, 0u);
    });
    _convert_image = device.compile<1>([this](BufferUInt accum, BufferFloat4 output) noexcept {
        auto i = dispatch_x();
        auto c0 = as<float>(accum.read(i * 4u + 0u));
        auto c1 = as<float>(accum.read(i * 4u + 1u));
        auto c2 = as<float>(accum.read(i * 4u + 2u));
        auto n = cast<float>(max(accum.read(i * 4u + 3u), 1u));
        auto scale = (1.f / n) * node<AtomicColorFilm>()->scale();
        output.write(i, make_float4(scale * make_float3(c0, c1, c2), 1.f));
    });
}

void AtomicColorFilmInstance::download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept {
    auto resolution = node()->resolution();
    auto pixel_count = resolution.x * resolution.y;
    command_buffer << _convert_image(_image, _converted).dispatch(pixel_count)
                   << _converted.copy_to(framebuffer);
}

void AtomicColorFilmInstance::accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept {
    $if(!any(isnan(rgb))) {
        auto pixel_id = pixel.y * node()->resolution().x + pixel.x;
        auto threshold = 16384.0f;
        auto lum = srgb_to_cie_y(rgb);
        auto c = rgb * (threshold / max(lum, threshold));
        for (auto i = 0u; i < 3u; i++) {
            atomic_float_add(_image, pixel_id * 4u + i, c[i]);
        }
        _image.atomic(pixel_id * 4u + 3u).fetch_add(1u);
    };
}

void AtomicColorFilmInstance::clear(CommandBuffer &command_buffer) noexcept {
    auto pixel_count = node()->resolution().x * node()->resolution().y;
    command_buffer << _clear_image(_image).dispatch(pixel_count);
}

luisa::unique_ptr<Film::Instance> AtomicColorFilm::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<AtomicColorFilmInstance>(pipeline.device(), pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::AtomicColorFilm)
