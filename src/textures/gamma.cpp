//
// Created by Mike Smith on 2022/1/26.
//

#include <core/thread_pool.h>
#include <util/imageio.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class GammaTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;
    float3 _gamma;

private:
    [[nodiscard]] float3 _v() const noexcept override { return _gamma; }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto color_gamma = pipeline.tex2d(handle->texture_id()).sample(uv).xyz();
        auto color = pow(color_gamma, handle->v());
        auto spec = pipeline.srgb_albedo_spectrum(color);
        return spec.sample(swl);
    }

public:
    GammaTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc},
          _gamma{desc->property_float3_or_default(
              "gamma", lazy_construct([desc] {
                  return make_float3(desc->property_float_or_default(
                      "gamma", 2.2f));
              }))} {
        auto path = desc->property_path("file");
        _img = ThreadPool::global().async([path = std::move(path)] {
            return LoadedImage::load(path, PixelStorage::BYTE4);
        });
        _gamma = clamp(_gamma, 1e-4f, 16.0f);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "gamma"; }
    [[nodiscard]] bool is_color() const noexcept override { return true; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GammaTexture)
