//
// Created by Mike Smith on 2022/1/26.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class ConstantGeneric final : public Texture {

private:
    float4 _v;
    uint _channels;

public:
    ConstantGeneric(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        auto v = desc->property_float_list_or_default("v");
        if (v.size() > 4u) [[unlikely]] {
            LUISA_WARNING(
                "Too many values (count = {}) for ConstValue. "
                "Additional values will be discarded. [{}]",
                v.size(), desc->source_location().string());
            v.resize(4u);
        }
        _channels = std::max(v.size(), static_cast<size_t>(1u));
        for (auto i = 0u; i < v.size(); i++) { _v[i] = v[i]; }
    }
    [[nodiscard]] auto v() const noexcept { return _v; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_v == 0.0f); }
    [[nodiscard]] bool is_constant() const noexcept override { return true; }
    [[nodiscard]] Category category() const noexcept override { return Category::GENERIC; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantGenericInstance final : public Texture::Instance {

private:
    luisa::optional<Differentiation::ConstantParameter> _diff_param;

public:
    ConstantGenericInstance(
        const Pipeline &p, const Texture *t,
        luisa::optional<Differentiation::ConstantParameter> param) noexcept
        : Texture::Instance{p, t}, _diff_param{std::move(param)} {}
    [[nodiscard]] Texture::Evaluation evaluate(const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto value = [&] {
            if (_diff_param) {
                return pipeline().differentiation().decode(*_diff_param);
            }
            return def(node<ConstantGeneric>()->v());
        }();
        return {.value = value, .scale = 1.f};
    }
    void backward(const Interaction &, const SampledWavelengths &, Expr<float>, Expr<float4> grad) const noexcept override {
        if (_diff_param) { pipeline().differentiation().accumulate(*_diff_param, grad); }
    }
};

luisa::unique_ptr<Texture::Instance> ConstantGeneric::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::optional<Differentiation::ConstantParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation().parameter(_v, _channels));
    }
    return luisa::make_unique<ConstantGenericInstance>(pipeline, this, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantGeneric)
