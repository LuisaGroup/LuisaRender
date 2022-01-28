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
    bool _is_black{};

private:
    [[nodiscard]] float3 _v() const noexcept override { return make_float3(); }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto rsp_scale = pipeline.tex2d(handle->texture_id()).sample(uv);
        auto scale = handle->v();
        auto spec = RGBIlluminantSpectrum{
            RGBSigmoidPolynomial{rsp_scale.xyz()}, rsp_scale.w,
            DenselySampledSpectrum::cie_illum_d6500()};
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
        _is_black = all(s <= 0.0f);
        auto half = desc->property_bool_or_default("half", false);
        _img = ThreadPool::global().async([path = std::move(path), half, s = max(s, 0.0f)] {
            if (half) {
                auto image = LoadedImage::load(path, PixelStorage::HALF4);
                auto pixels = reinterpret_cast<std::array<uint16_t, 4u> *>(image.pixels());
                for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                    auto [x, y, z, _] = pixels[i];
                    auto f = make_float3(half_to_float(x), half_to_float(y), half_to_float(z));
                    auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(f * s);
                    pixels[i] = {static_cast<uint16_t>(float_to_half(rsp.x)),
                                 static_cast<uint16_t>(float_to_half(rsp.y)),
                                 static_cast<uint16_t>(float_to_half(rsp.z)),
                                 static_cast<uint16_t>(float_to_half(scale))};
                }
                return image;
            }
            auto image = LoadedImage::load(path, PixelStorage::FLOAT4);
            auto pixels = reinterpret_cast<float4 *>(image.pixels());
            for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                auto p = pixels[i].xyz() * s;
                auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(p);
                pixels[i] = make_float4(rsp, scale);
            }
            return image;
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdrillum"; }
    [[nodiscard]] bool is_color() const noexcept override { return false; }
    [[nodiscard]] bool is_generic() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return true; }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HDRIlluminantTexture)
