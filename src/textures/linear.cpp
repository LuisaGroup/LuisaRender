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
    std::shared_future<LoadedImage> _image;

private:
    [[nodiscard]] std::pair<uint, float3> _encode(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto texture_id = pipeline.image_texture(command_buffer, _image.get(), sampler());
        return std::make_pair(texture_id, make_float3());
    }
    [[nodiscard]] Float4 _evaluate(
        const Pipeline &pipeline, const Var<TextureHandle> &handle,
        Expr<float2> uv, const SampledWavelengths &swl) const noexcept override {
        auto linear = pipeline.tex2d(handle->texture_id()).sample(uv);
        auto spec = pipeline.srgb_albedo_spectrum(linear.xyz());
        return spec.sample(swl);
    }

public:
    LinearTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc} {
        auto path = desc->property_path("file");
        auto half = desc->property_bool_or_default("half", false);
        _image = ThreadPool::global().async([path = std::move(path), half] {
            return LoadedImage::load(path, half ? PixelStorage::HALF4 : PixelStorage::FLOAT4);
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdrillum"; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_color() const noexcept override { return true; }
    [[nodiscard]] bool is_value() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LinearTexture)
