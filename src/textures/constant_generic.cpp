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
    [[nodiscard]] Category category() const noexcept override { return Category::GENERIC; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

struct ConstantGenericInstance final : public Texture::Instance {
    ConstantGenericInstance(const Pipeline &p, const Texture *t) noexcept
        : Texture::Instance{p, t} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        return node<ConstantGeneric>()->v();
    }
};

luisa::unique_ptr<Texture::Instance> ConstantGeneric::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ConstantGenericInstance>(pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantGeneric)
