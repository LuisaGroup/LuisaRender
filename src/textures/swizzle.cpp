//
// Created by Mike Smith on 2022/10/4.
//

#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class SwizzleTexture final : public Texture {

private:
    Texture *_base;
    uint _swizzle{};

public:
    SwizzleTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _base{scene->load_texture(desc->property_node("base"))} {
        auto swizzle = desc->property_uint_list_or_default(
            "swizzle", lazy_construct([&] {
                using namespace std::string_view_literals;
                auto s = desc->property_string_or_default("swizzle", "rgba");
                luisa::vector<uint> swizzle_channels;
                for (auto c : s) {
                    switch (c) {
                        case 'r': [[fallthrough]];
                        case 'x': swizzle_channels.push_back(0u); break;
                        case 'g': [[fallthrough]];
                        case 'y': swizzle_channels.push_back(1u); break;
                        case 'b': [[fallthrough]];
                        case 'z': swizzle_channels.push_back(2u); break;
                        case 'a': [[fallthrough]];
                        case 'w': swizzle_channels.push_back(3u); break;
                        default: LUISA_ERROR_WITH_LOCATION(
                            "Invalid swizzle channel '{}'. [{}]",
                            c, desc->source_location().string());
                    }
                }
                return swizzle_channels;
            }));
        if (swizzle.size() > 4u) [[unlikely]] {
            LUISA_WARNING(
                "Too many swizzle channels (count = {}) for SwizzleTexture. "
                "Additional channels will be discarded. [{}]",
                swizzle.size(), desc->source_location().string());
            swizzle.resize(4u);
        }
        for (auto i = 0u; i < swizzle.size(); i++) {
            auto c = swizzle[i];
            LUISA_ASSERT(c < 4u, "Swizzle channel '{}' out of range. [{}]",
                         c, desc->source_location().string());
            _swizzle |= c << (i * 4u);
        }
        _swizzle |= static_cast<uint>(swizzle.size()) << 16u;
    }
    [[nodiscard]] auto base() const noexcept { return _base; }
    [[nodiscard]] auto swizzle(uint i) const noexcept {
        LUISA_ASSERT(i < channels(), "Swizzle channel index out of range.");
        return (_swizzle >> (i * 4u)) & 0b1111u;
    }
    [[nodiscard]] bool is_black() const noexcept override { return _base->is_black(); }
    [[nodiscard]] bool is_constant() const noexcept override { return _base->is_constant(); }
    [[nodiscard]] luisa::optional<float4> evaluate_static() const noexcept override {
        if (auto v = _base->evaluate_static()) {
            auto s = make_float4(0.f);
            for (auto i = 0u; i < channels(); i++) { s[i] = (*v)[swizzle(i)]; }
            return s;
        }
        return nullopt;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _swizzle >> 16u; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] bool requires_gradients() const noexcept override { return _base->requires_gradients(); }
    void disable_gradients() noexcept override { _base->disable_gradients(); }
};

class SwizzleTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_base;

public:
    SwizzleTextureInstance(const Pipeline &pipeline, const Texture *node,
                           const Texture::Instance *base) noexcept
        : Texture::Instance{pipeline, node}, _base{base} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  const SampledWavelengths &swl,
                                  Expr<float> time) const noexcept override {
        auto v = _base->evaluate(it, swl, time);
        switch (auto n = node<SwizzleTexture>(); n->channels()) {
            case 1u: return make_float4(v[n->swizzle(0u)]);
            case 2u: return make_float4(v[n->swizzle(0u)], v[n->swizzle(1u)], 0.0f, 1.0f);
            case 3u: return make_float4(v[n->swizzle(0u)], v[n->swizzle(1u)], v[n->swizzle(2u)], 1.0f);
            case 4u: return make_float4(v[n->swizzle(0u)], v[n->swizzle(1u)], v[n->swizzle(2u)], v[n->swizzle(3u)]);
            default: LUISA_ERROR_WITH_LOCATION("Unreachable");
        }
        return make_float4();
    }
    void backward(const Interaction &it, const SampledWavelengths &swl, Expr<float> time, Expr<float4> grad) const noexcept override {
        if (node()->requires_gradients()) {
            auto g = def(make_float4());
            auto n = node<SwizzleTexture>();
            for (auto i = 0u; i < n->channels(); i++) {
                auto c = n->swizzle(i);
                g[c] += grad[i];
            }
            _base->backward(it, swl, time, g);
        }
    }
    [[nodiscard]] luisa::string diff_param_identifier() const noexcept override {
        auto base_ident = Instance::diff_param_identifier(_base);
        return base_ident == non_differrentiable_identifier ?
                   non_differrentiable_identifier :
                   luisa::format("diffswizzle<{}, {}>",
                                 base_ident, node<SwizzleTexture>()->channels());
    }
};

luisa::unique_ptr<Texture::Instance> SwizzleTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto base = pipeline.build_texture(command_buffer, _base);
    return luisa::make_unique<SwizzleTextureInstance>(pipeline, this, base);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SwizzleTexture)
