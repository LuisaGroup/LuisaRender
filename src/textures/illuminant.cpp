//
// Created by Mike Smith on 2022/1/28.
//

#include <core/clock.h>
#include <core/thread_pool.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class IlluminantTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;
    bool _is_black{};

private:
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }

public:
    IlluminantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc} {
        auto path = desc->property_path("file");
        auto fp32 = desc->property_bool_or_default("fp32", false);
        auto encoding = desc->property_string_or_default(
            "encoding", lazy_construct([&path]() noexcept -> luisa::string {
                auto ext = path.extension().string();
                for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
                if (ext == ".exr" || ext == ".hdr") { return "linear"; }
                return "sRGB";
            }));
        for (auto &c : encoding) { c = static_cast<char>(tolower(c)); }
        auto gamma = 1.0f;
        if (encoding == "gamma") { gamma = desc->property_float_or_default("gamma", 2.2f); }
        auto scale = desc->property_float3_or_default(
            "scale", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "scale", 1.0f));
            }));
        scale = clamp(scale, 0.0f, 1024.0f);
        _is_black = all(scale == 0.0f);
        _img = ThreadPool::global().async([path = std::move(path), half = !fp32,
                                           encoding = std::move(encoding), gamma,
                                           sloc = desc->source_location(), scale] {
            auto image = LoadedImage::load(path, half ? PixelStorage::HALF4 : PixelStorage::FLOAT4);
            if (encoding == "rsp") { return image; }
            auto process = [&]() -> luisa::function<float4(float3)> {
                auto rgb2spec = [scale](auto p) noexcept {
                    auto rsp_scale = RGB2SpectrumTable::srgb().decode_unbound(p * scale);
                    return make_float4(rsp_scale.first, rsp_scale.second);
                };
                if (encoding == "linear") { return rgb2spec; }
                if (encoding == "srgb") {
                    return [rgb2spec](auto p) noexcept {
                        auto s2l = [](auto x) noexcept {
                            return x <= 0.04045f ?
                                       x * (1.0f / 12.92f) :
                                       std::pow((x + 0.055f) * (1.0f / 1.055f), 2.4f);
                        };
                        return rgb2spec(make_float3(s2l(p.x), s2l(p.y), s2l(p.z)));
                    };
                }
                if (encoding == "gamma") {
                    return [rgb2spec, gamma](auto p) noexcept {
                        auto g = [gamma](auto x) noexcept { return std::pow(x, gamma); };
                        return rgb2spec(make_float3(g(p.x), g(p.y), g(p.z)));
                    };
                }
                LUISA_ERROR(
                    "Unknown color texture encoding '{}'. [{}]",
                    encoding, sloc.string());
            }();
            if (half) {
                auto pixels = reinterpret_cast<std::array<uint16_t, 4u> *>(image.pixels());
                for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                    auto [x, y, z, _] = pixels[i];
                    auto f = make_float3(half_to_float(x), half_to_float(y), half_to_float(z));
                    auto rsp = process(f);
                    pixels[i][0] = float_to_half(rsp.x);
                    pixels[i][1] = float_to_half(rsp.y);
                    pixels[i][2] = float_to_half(rsp.z);
                    pixels[i][3] = float_to_half(rsp.w);
                }
            } else {
                auto pixels = reinterpret_cast<float4 *>(image.pixels());
                for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                    pixels[i] = process(pixels[i].xyz());
                }
            }
            return image;
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] Category category() const noexcept override { return Category::ILLUMINANT; }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::IlluminantTexture)
