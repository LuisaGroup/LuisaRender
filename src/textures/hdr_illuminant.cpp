//
// Created by Mike Smith on 2022/1/26.
//

#include <core/clock.h>
#include <core/thread_pool.h>
#include <util/imageio.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class HDRIlluminantTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _image;
    float3 _scale;

private:
    [[nodiscard]] std::pair<uint, float3> _encode(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto texture_id = pipeline.image_texture(command_buffer, _image.get(), sampler());
        return std::make_pair(texture_id, make_float3());
    }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto rsp_scale = pipeline.tex2d(handle->texture_id()).sample(uv);
        RGBIlluminantSpectrum spec{
            RGBSigmoidPolynomial{rsp_scale.xyz()}, rsp_scale.w,
            DenselySampledSpectrum::cie_illum_d6500()};
        return spec.sample(swl);
    }

public:
    HDRIlluminantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc},
          _scale{max(
              desc->property_float3_or_default(
                  "scale", lazy_construct([desc] {
                      return make_float3(desc->property_float_or_default(
                          "scale", 1.0f));
                  })),
              0.0f)} {
        auto path = desc->property_path("file");
        _image = ThreadPool::global().async([path = std::move(path), s = _scale] {
            auto image = LoadedImage::load(path, PixelStorage::FLOAT4);
            for (auto i = 0u; i < image.size().x * image.size().y; i++) {
                auto &p = reinterpret_cast<float4 *>(image.pixels())[i];
                auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(p.xyz() * s);
                p = make_float4(rsp, scale);
            }
            return image;
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
