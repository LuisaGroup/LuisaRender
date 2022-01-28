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
    float3 _scale;

private:
    [[nodiscard]] float3 _v() const noexcept override { return _scale; }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto scale = handle->v();
        auto L = pipeline.tex2d(handle->texture_id()).sample(uv);
        auto spec = pipeline.srgb_illuminant_spectrum(L.xyz() * scale);
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
        _scale = max(s, 0.0f);
        auto half = desc->property_bool_or_default("half", false);
        _img = ThreadPool::global().async([path = std::move(path), half] {
            return LoadedImage::load(path, half ? PixelStorage::HALF4 : PixelStorage::FLOAT4);
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdrillum"; }
    [[nodiscard]] bool is_color() const noexcept override { return false; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return true; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_scale == 0.0f); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HDRIlluminantTexture)
