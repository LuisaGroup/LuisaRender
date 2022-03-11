//
// Created by Mike on 2022/1/7.
//

#include <tinyexr.h>

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
    Shader1D<Buffer<uint>> _clear_image;

public:
    AtomicColorFilmInstance(Device &device, Pipeline &pipeline, const AtomicColorFilm *film) noexcept;
    void accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept override;
    void save(Stream &stream, const std::filesystem::path &path) const noexcept override;
    void clear(CommandBuffer &command_buffer) noexcept override;
};

AtomicColorFilmInstance::AtomicColorFilmInstance(Device &device, Pipeline &pipeline, const AtomicColorFilm *film) noexcept
    : Film::Instance{pipeline, film} {
    auto resolution = node()->resolution();
    auto pixel_count = resolution.x * resolution.y;
    _image = pipeline.device().create_buffer<uint>(pixel_count * 4u);
    Kernel1D clear_image = [](BufferUInt image) noexcept {
        image.write(dispatch_x() * 4u + 0u, 0u);
        image.write(dispatch_x() * 4u + 1u, 0u);
        image.write(dispatch_x() * 4u + 2u, 0u);
        image.write(dispatch_x() * 4u + 3u, 0u);
    };
    _clear_image = device.compile(clear_image);
}

void AtomicColorFilmInstance::save(Stream &stream, const std::filesystem::path &path) const noexcept {
    auto resolution = node()->resolution();
    auto file_ext = path.extension().string();
    for (auto &c : file_ext) { c = static_cast<char>(tolower(c)); }
    std::vector<float4> rgb(resolution.x * resolution.y);
    stream << _image.copy_to(rgb.data()) << synchronize();
    auto film = static_cast<const AtomicColorFilm *>(node());
    auto scale = film->scale();
    for (auto i = 0; i < resolution.x * resolution.y; i++) {
        auto n = luisa::bit_cast<uint>(rgb[i].w);
        auto v = 1.f / static_cast<float>(n) * scale * rgb[i].xyz();
        rgb[i] = make_float4(v, 1.f);
    }
    if (file_ext == ".exr") {
        const char *err = nullptr;
        auto ret = SaveEXR(
            reinterpret_cast<const float *>(rgb.data()),
            static_cast<int>(resolution.x), static_cast<int>(resolution.y),
            4, false, path.string().c_str(), &err);
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

void AtomicColorFilmInstance::accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept {
    $if (!any(isnan(rgb))) {
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
