//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>
#include <util/rng.h>
#include <util/scattering.h>
#include <base/spectrum.h>

namespace luisa::render {

class FresnelTexture final : public Texture {

private:
    const Texture *_eta;

public:
    FresnelTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _eta{scene->load_texture(desc->property_node_or_default("eta"))} {
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(FresnelTexture, eta, 1);
        LUISA_ASSERT(semantic() == Texture::Semantic::GENERIC,
                     "FresnelTexture can only be used as generic textures.");
    }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_constant() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return 1u; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] bool requires_gradients() const noexcept override { return _eta != nullptr && _eta->requires_gradients(); }
};

class FresnelTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_eta;

public:
    FresnelTextureInstance(const Pipeline &p, const Texture *t,
                           const Texture::Instance *eta) noexcept
        : Texture::Instance{p, t}, _eta{eta} {}
    [[nodiscard]] Float4 evaluate(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto eta = _eta == nullptr ? def(1.5f) : _eta->evaluate(it, swl, time).x;
        auto R0 = sqr((eta - 1.f) / (eta + 1.f));
        auto cos_theta = abs_cos_theta(it.wo_local());
        auto m = saturate(1.f - cos_theta);
        auto m5 = sqr(sqr(m)) * m;
        return make_float4(lerp(R0, 1.f, m5));
    }
    void backward(const Interaction &it, const SampledWavelengths &swl,
                  Expr<float> time, Expr<float4> grad) const noexcept override {
        if (node()->requires_gradients()) {
            auto cos_theta = abs_cos_theta(it.wo_local());
            auto m = saturate(1.f - cos_theta);
            auto m5 = sqr(sqr(m)) * m;
            auto dR0 = 1.f - m5;
            auto f = lerp(dR0, 1.f, m5);
            auto cubic = [](auto x) noexcept { return x * x * x; };
            auto dEta = -4.f * (1 - f) / cubic(1 + f) * dR0;
            _eta->backward(it, swl, time, grad * dEta);
        }
    }
};

luisa::unique_ptr<Texture::Instance> FresnelTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto eta = pipeline.build_texture(command_buffer, _eta);
    return luisa::make_unique<FresnelTextureInstance>(pipeline, this, eta);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::FresnelTexture)
