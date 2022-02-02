//
// Created by Mike Smith on 2022/1/12.
//

#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class MirrorSurface final : public Surface {

private:
    const Texture *_color;

public:
    MirrorSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", SceneNodeDesc::shared_default_texture("ConstColor")))} {
        if (_color->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in MirrorSurface. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<TextureHandle>(1u);
        auto texture_handle = pipeline.encode_texture(command_buffer, _color);
        command_buffer << buffer_view.copy_from(texture_handle);
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class MirrorClosure final : public Surface::Closure {

private:
    const Interaction &_it;
    const SampledWavelengths &_swl;
    Float4 _refl;

public:
    MirrorClosure(const Interaction &it, const SampledWavelengths &swl, Expr<float4> refl) noexcept
        : _it{it}, _swl{swl}, _refl{refl} {}
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        return {.swl = _swl, .f = make_float4(0.0f), .pdf = 0.0f};
    }
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto cos_wo = dot(_it.wo(), _it.shading().n());
        auto wi = 2.0f * cos_wo * _it.shading().n() - _it.wo();
        static constexpr auto delta_pdf = 1e8f;
        Surface::Evaluation eval{
            .swl = _swl,
            .f = delta_pdf * _refl / cos_wo,
            .pdf = ite(cos_wo > 0.0f, delta_pdf, 0.0f)};
        return {.wi = std::move(wi), .eval = std::move(eval)};
    }
};

unique_ptr<Surface::Closure> MirrorSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto texture = pipeline.buffer<TextureHandle>(it.shape()->surface_buffer_id()).read(0u);
    auto R = pipeline.evaluate_color_texture(texture, it, swl, time);
    return luisa::make_unique<MirrorClosure>(it, swl, std::move(R));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MirrorSurface)
