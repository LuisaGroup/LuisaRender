//
// Created by Mike Smith on 2022/1/26.
//

#include <core/thread_pool.h>
#include <util/imageio.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class SRGBIlluminantTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;
    float3 _scale;

private:
    [[nodiscard]] float3 _v() const noexcept override { return _scale; }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto color_srgb = pipeline.tex2d(handle->texture_id()).sample(uv).xyz();
        auto srgb2linear = [](Expr<float3> x) noexcept {
            return ite(
                x <= 0.04045f,
                x * (1.0f / 12.92f),
                pow((x + 0.055f) * (1.0f / 1.055f), 2.4f));
        };
        auto color = srgb2linear(color_srgb);
        auto spec = pipeline.srgb_illuminant_spectrum(color * handle->v());
        return spec.sample(swl);
    }

public:
    SRGBIlluminantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc},
          _scale{max(
              desc->property_float3_or_default(
                  "scale", lazy_construct([desc] {
                      return make_float3(desc->property_float_or_default(
                          "scale", 1.0f));
                  })),
              0.0f)} {
        auto path = desc->property_path("file");
        _img = ThreadPool::global().async([path = std::move(path)] {
            return LoadedImage::load(path, PixelStorage::BYTE4);
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "srgbillum"; }
    [[nodiscard]] bool is_color() const noexcept override { return false; }
    [[nodiscard]] bool is_generic() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return true; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_scale == 0.0f); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SRGBIlluminantTexture)
