//
// Created by Mike Smith on 2022/1/26.
//

#include <core/thread_pool.h>
#include <util/imageio.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class SRGBTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage<uint8_t>> _image;

private:
    [[nodiscard]] std::pair<uint, float3> _encode(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto texture_id = pipeline.image_texture(command_buffer, _image.get(), sampler());
        return std::make_pair(texture_id, make_float3());
    }
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
        auto spec = pipeline.srgb_albedo_spectrum(color);
        return spec.sample(swl);
    }

public:
    SRGBTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc} {
        auto path = desc->property_path("file");
        _image = ThreadPool::global().async([path = std::move(path)] {
            return load_ldr_image(path, 4u);
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "srgb"; }
    [[nodiscard]] bool is_color() const noexcept override { return true; }
    [[nodiscard]] bool is_general() const noexcept override { return false; }
    [[nodiscard]] bool is_illuminant() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SRGBTexture)
