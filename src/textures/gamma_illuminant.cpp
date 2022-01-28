//
// Created by Mike Smith on 2022/1/26.
//

#include <core/thread_pool.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class GammaIlluminantTexture final : public ImageTexture {

public:
    static constexpr auto gamma_min = 1e-3f;
    static constexpr auto gamma_max = 15.0f;
    static constexpr auto gamma_scale = 1024.0f;

private:
    std::shared_future<LoadedImage> _img;
    float3 _gamma;
    float3 _scale;

private:
    [[nodiscard]] float3 _v() const noexcept override {
        auto fixed_point_gamma = [](auto f) noexcept {
            return static_cast<uint>(
                std::round(f * gamma_scale));
        };
        auto v = make_uint3(
            (fixed_point_gamma(_gamma.x) << 16u) | float_to_half(_scale.x),
            (fixed_point_gamma(_gamma.y) << 16u) | float_to_half(_scale.y),
            (fixed_point_gamma(_gamma.z) << 16u) | float_to_half(_scale.z));
        return luisa::bit_cast<float3>(v);
    }
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto v = as<uint3>(handle->v());
        auto gv = v >> 16u;
        auto sv = v & 0xffffu;
        auto gamma = make_float3(gv) * (1.0f / gamma_scale);
        auto scale = make_float3(half_to_float(sv.x), half_to_float(sv.y), half_to_float(sv.z));
        auto color_gamma = pipeline.tex2d(handle->texture_id()).sample(uv).xyz();
        auto color = pow(color_gamma, gamma);
        auto spec = pipeline.srgb_illuminant_spectrum(color * scale);
        return spec.sample(swl);
    }

public:
    GammaIlluminantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc},
          _gamma{desc->property_float3_or_default(
              "gamma", lazy_construct([desc] {
                  return make_float3(desc->property_float_or_default(
                      "gamma", 2.2f));
              }))},
          _scale{desc->property_float3_or_default(
              "scale", lazy_construct([desc] {
                  return make_float3(desc->property_float_or_default(
                      "scale", 1.0f));
              }))} {
        auto path = desc->property_path("file");
        _img = ThreadPool::global().async([path = std::move(path)] {
            return LoadedImage::load(path, PixelStorage::BYTE4);
        });
        _gamma = clamp(_gamma, gamma_min, gamma_max);
        _scale = clamp(_scale, 1.0f / 1024.0f, 1024.0f);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "gammaillum"; }
    [[nodiscard]] bool is_color() const noexcept override { return false; }
    [[nodiscard]] bool is_generic() const noexcept override { return false; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_scale == 0.0f); }
    [[nodiscard]] bool is_illuminant() const noexcept override { return true; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GammaIlluminantTexture)
