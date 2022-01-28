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

class LinearTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;

private:
    [[nodiscard]] float3 _v() const noexcept override { return make_float3(0.0f); }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto rsp = pipeline.tex2d(handle->texture_id()).sample(uv).xyz();
        auto spec = RGBAlbedoSpectrum{RGBSigmoidPolynomial{rsp}};
        return spec.sample(swl);
    }

public:
    LinearTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc} {
        auto path = desc->property_path("file");
        auto half = desc->property_bool_or_default("half", false);
        _img = ThreadPool::global().async([path = std::move(path), half] {
            if (half) {
                auto image = LoadedImage::load(path, PixelStorage::HALF4);
                auto pixels = reinterpret_cast<std::array<uint16_t, 4u> *>(image.pixels());
                for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                    auto [x, y, z, _] = pixels[i];
                    auto f = make_float3(half_to_float(x), half_to_float(y), half_to_float(z));
                    auto rsp = RGB2SpectrumTable::srgb().decode_albedo(f);
                    pixels[i][0] = float_to_half(rsp.x);
                    pixels[i][1] = float_to_half(rsp.y);
                    pixels[i][2] = float_to_half(rsp.z);
                }
                return image;
            }
            auto image = LoadedImage::load(path, PixelStorage::FLOAT4);
            auto pixels = reinterpret_cast<float4 *>(image.pixels());
            for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                auto p = pixels[i];
                auto rsp = RGB2SpectrumTable::srgb().decode_albedo(p.xyz());
                pixels[i] = make_float4(rsp, p.w);
            }
            return image;
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "linear"; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_color() const noexcept override { return true; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LinearTexture)
