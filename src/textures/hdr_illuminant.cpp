//
// Created by Mike Smith on 2022/1/26.
//

#include <core/clock.h>
#include <core/thread_pool.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class HDRIlluminantTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;
    std::array<float, 3u> _scale{};
    bool _half{};
    bool _is_black{};

private:
    [[nodiscard]] float3 _v() const noexcept override {
        return _half ?
                   make_float3(_scale[0], _scale[1], _scale[2]) :
                   make_float3(-1.0f);
    }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto L = pipeline.tex2d(handle->texture_id()).sample(uv);
        auto scale = handle->v();
        auto spec = RGBIlluminantSpectrum{
            RGBSigmoidPolynomial{L.xyz()}, L.w,
            DenselySampledSpectrum::cie_illum_d6500()};
        $if (scale.x >= 0.0f) {
            spec = pipeline.srgb_illuminant_spectrum(
                L.xyz() * scale);
        };
        return spec.sample(swl);
    }

public:
    HDRIlluminantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc} {
        auto path = desc->property_path("file");
        auto s = desc->property_float3_or_default(
            "scale", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default(
                    "scale", 1.0f));
            }));
        s = max(s, 0.0f);
        _scale = {s.x, s.y, s.z};
        _is_black = all(s == 0.0f);
        _half = desc->property_bool_or_default("half", false);
        _img = ThreadPool::global().async([path = std::move(path), half = _half, s] {
            if (half) { return LoadedImage::load(path, PixelStorage::HALF4); }
            auto image = LoadedImage::load(path, PixelStorage::FLOAT4);
            auto pixels = reinterpret_cast<float4 *>(image.pixels());
            for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                auto p = pixels[i];
                auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(p.xyz() * s);
                pixels[i] = make_float4(rsp, scale);
            }
            return image;
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdrillum"; }
    [[nodiscard]] bool is_color() const noexcept override { return false; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return true; }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HDRIlluminantTexture)
