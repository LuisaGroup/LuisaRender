//
// Created by Mike Smith on 2022/11/9.
//

#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class CheckerboardTexture final : public Texture {

private:
    Texture *_on;
    Texture *_off;
    float2 _scale;

public:
    CheckerboardTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _on{scene->load_texture(desc->property_node_or_default("on"))},
          _off{scene->load_texture(desc->property_node_or_default("off"))},
          _scale{desc->property_float2_or_default(
              "scale", lazy_construct([desc] {
                  return make_float2(desc->property_float_or_default("scale", 1.0f));
              }))} {}
    [[nodiscard]] bool is_black() const noexcept override {
        auto on_is_black = _on != nullptr && _on->is_black();   // on is default to all white
        auto off_is_black = _off == nullptr || _off->is_black();// off is default to all black
        return on_is_black && off_is_black;
    }
    [[nodiscard]] bool is_constant() const noexcept override {
        auto on_is_constant = _on == nullptr || _on->is_constant();
        auto off_is_constant = _off == nullptr || _off->is_constant();
        return on_is_constant && off_is_constant;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override {
        auto on_channels = _on == nullptr ? 4u : _on->channels();
        auto off_channels = _off == nullptr ? 4u : _off->channels();
        if (on_channels != off_channels) {
            LUISA_WARNING_WITH_LOCATION(
                "CheckerboardTexture: on and off textures "
                "have different channel counts ({} vs {}).",
                on_channels, off_channels);
        }
        return std::min(on_channels, off_channels);
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class CheckerboardTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_on;
    const Texture::Instance *_off;

private:
    [[nodiscard]] auto _select(Expr<float2> uv) const noexcept {
        auto t = uv * node<CheckerboardTexture>()->scale();
        return (cast<int>(floor(t.x)) + cast<int>(floor(t.y))) % 2 == 0;
    }

public:
    CheckerboardTextureInstance(const Pipeline &pipeline, const Texture *node,
                                const Texture::Instance *on, const Texture::Instance *off) noexcept
        : Texture::Instance{pipeline, node}, _on{on}, _off{off} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  const SampledWavelengths &swl,
                                  Expr<float> time) const noexcept override {
        auto value = def(make_float4());
        $if(_select(it.uv())) {
            value = _on == nullptr ? make_float4(1.f) : _on->evaluate(it, swl, time);
        }
        $else {
            value = _off == nullptr ? make_float4(0.f) : _off->evaluate(it, swl, time);
        };
        return value;
    }
    [[nodiscard]] Spectrum::Decode evaluate_albedo_spectrum(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        Spectrum::Decode value{SampledSpectrum{swl.dimension()}, 0.f};
        $if(_select(it.uv())) {
            value = _on == nullptr ?
                        Spectrum::Decode{SampledSpectrum{swl.dimension()}, 1.f} :
                        _on->evaluate_albedo_spectrum(it, swl, time);
        }
        $else {
            value = _off == nullptr ?
                        Spectrum::Decode{SampledSpectrum{swl.dimension()}, 0.f} :
                        _off->evaluate_albedo_spectrum(it, swl, time);
        };
        return value;
    }
    [[nodiscard]] Spectrum::Decode evaluate_illuminant_spectrum(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        Spectrum::Decode value{SampledSpectrum{swl.dimension()}, 0.f};
        $if(_select(it.uv())) {
            value = _on == nullptr ?
                        Spectrum::Decode{SampledSpectrum{swl.dimension()}, 1.f} :
                        _on->evaluate_illuminant_spectrum(it, swl, time);
        }
        $else {
            value = _off == nullptr ?
                        Spectrum::Decode{SampledSpectrum{swl.dimension()}, 0.f} :
                        _off->evaluate_illuminant_spectrum(it, swl, time);
        };
        return value;
    }
};

luisa::unique_ptr<Texture::Instance> CheckerboardTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto on = pipeline.build_texture(command_buffer, _on);
    auto off = pipeline.build_texture(command_buffer, _off);
    return luisa::make_unique<CheckerboardTextureInstance>(pipeline, this, on, off);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::CheckerboardTexture)
