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

class ColorTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;
    bool _is_black{};

private:
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }

public:
    ColorTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
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
        auto gamma = make_float3(1.0f);
        if (encoding == "gamma") {
            gamma = desc->property_float3_or_default(
                "gamma", lazy_construct([desc] {
                    return make_float3(desc->property_float_or_default(
                        "gamma", 2.2f));
                }));
        }
        gamma = clamp(gamma, 1e-3f, 16.0f);
        auto tint = desc->property_float3_or_default(
            "tint", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "tint", 1.0f));
            }));
        tint = max(tint, 0.0f);
        _is_black = all(tint == 0.0f);
        _img = ThreadPool::global().async([path = std::move(path), half = !fp32, encoding = std::move(encoding),
                                           gamma, tint, sloc = desc->source_location()] {
            auto image = LoadedImage::load(path, half ? PixelStorage::HALF4 : PixelStorage::FLOAT4);
            if (encoding == "rsp") { return image; }
            auto process = [&]() -> luisa::function<float3(float3)> {
                auto rgb2spec = [tint](auto p) noexcept {
                    auto rsp = RGB2SpectrumTable::srgb().decode_albedo(p * tint);
                    return make_float3(rsp.x, rsp.y, rsp.z);
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
                    return [rgb2spec, g = gamma](auto p) noexcept {
                        return rgb2spec(make_float3(
                            pow(p.x, g.x), pow(p.y, g.y), pow(p.z, g.z)));
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
                }
            } else {
                auto pixels = reinterpret_cast<float4 *>(image.pixels());
                for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                    auto p = pixels[i];
                    auto rsp = process(p.xyz());
                    pixels[i] = make_float4(rsp, p.w);
                }
            }
            return image;
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "color"; }
    [[nodiscard]] Category category() const noexcept override { return Category::COLOR; }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ColorTexture)
