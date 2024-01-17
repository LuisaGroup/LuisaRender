//
// Created by Mike Smith on 2022/1/26.
//

#include "util/spec.h"
#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>
#include <util/rng.h>

namespace luisa::render {

class ConstantTexture final : public Texture {

private:
    float4 _v;
    uint _channels{0u};
    bool _black{false};
    bool _should_inline;

public:
    ConstantTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        if (requires_gradients())
            _should_inline = false;
        else
            _should_inline = desc->property_bool_or_default("inline", true);
        auto scale = desc->property_float_or_default("scale", 1.f);
        auto v = desc->property_float_list_or_default("v");
        if (v.empty()) [[unlikely]] {
            LUISA_WARNING(
                "No value for ConstantTexture. "
                "Fallback to single-channel zero. [{}]",
                desc->source_location().string());
            v.emplace_back(0.f);
        } else if (v.size() > 4u) [[unlikely]] {
            LUISA_WARNING(
                "Too many values (count = {}) for ConstantTexture. "
                "Additional values will be discarded. [{}]",
                v.size(), desc->source_location().string());
            v.resize(4u);
        }
        _channels = v.size();
        for (auto i = 0u; i < v.size(); i++) { _v[i] = scale * v[i]; }
        _black = all(_v == 0.f);
    }
    [[nodiscard]] auto v() const noexcept { return _v; }
    [[nodiscard]] bool is_black() const noexcept override { return _black; }
    [[nodiscard]] bool is_constant() const noexcept override { return true; }
    [[nodiscard]] bool should_inline() const noexcept { return _should_inline; }
    [[nodiscard]] optional<float4> evaluate_static() const noexcept override {
        return _should_inline ? luisa::make_optional(_v) : luisa::nullopt;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantTextureInstance final : public Texture::Instance {

private:
    uint _constant_slot{};
    luisa::optional<Differentiation::ConstantParameter> _diff_param;

public:
    ConstantTextureInstance(Pipeline &p,
                            const ConstantTexture *t,
                            CommandBuffer &cmd_buffer,
                            luisa::optional<Differentiation::ConstantParameter> param) noexcept
        : Texture::Instance{p, t}, _diff_param{std::move(param)} {
        if (!t->should_inline()) {
            auto [buffer, buffer_id] = p.allocate_constant_slot();
            auto v = t->v();
            cmd_buffer << buffer.copy_from(&v) << compute::commit();
            _constant_slot = buffer_id;
        }
    }
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  const SampledWavelengths &swl,
                                  Expr<float> time) const noexcept override {
        if (_diff_param) { return pipeline().differentiation()->decode(*_diff_param); }
        if (auto texture = node<ConstantTexture>();
            texture->should_inline()) { return texture->v(); }
        return pipeline().constant(_constant_slot);
    }
    [[nodiscard]] SampledSpectrum eval_grad(const Interaction &it,
                                            const SampledWavelengths &swl,
                                            Expr<float> time,
                                            Expr<float4> grad) const noexcept override {
        if (_diff_param) {
            if (node()->render_grad_map()) {
                // render grad map in a simple way:
                // if texture param 3-dim, combine them. which means: add up each dim of grad, only eval_grad qualitatively.
                // as for output: each dim of the sampledspectrum will be the same.
                auto grads = (grad[0] + grad[1] + grad[2]);
                return {swl.dimension(), ite(isnan(grads), 0.f, grads)};
            } else {
                return {swl.dimension(), 0.f};
            }
        } else {
            return {swl.dimension(), 0.f};
        }
    }
    [[nodiscard]] Spectrum::Decode evaluate_albedo_spectrum(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (_diff_param) { return Instance::evaluate_albedo_spectrum(it, swl, time); }
        auto tex = node<ConstantTexture>();
        auto spec = pipeline().spectrum();
        auto enc = spec->node()->encode_srgb_albedo(tex->v().xyz());
        return spec->decode_albedo(swl, enc);
    }
    [[nodiscard]] Spectrum::Decode evaluate_illuminant_spectrum(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (_diff_param) { return Instance::evaluate_illuminant_spectrum(it, swl, time); }
        auto tex = node<ConstantTexture>();
        auto spec = pipeline().spectrum();
        auto enc = spec->node()->encode_srgb_illuminant(tex->v().xyz());
        return spec->decode_illuminant(swl, enc);
    }
    void backward(const Interaction &it, const SampledWavelengths &swl,
                  Expr<float> time, Expr<float4> grad) const noexcept override {
        if (_diff_param) {
            // $if(_diff_param->index() == 1u) {
            // $if(grad[0] > 10.f) {
            //     device_log("diff param ({}) , grad in accumulate: ({}, {}, {})",
            //                _diff_param->index(), grad[0u], grad[1u], grad[2u]);
            // };
            auto slot_seed = xxhash32(as<uint3>(it.p()));
            pipeline().differentiation()->accumulate(*_diff_param, grad, slot_seed);
        }
    }
    [[nodiscard]] luisa::string diff_param_identifier() const noexcept override {
        return _diff_param ? _diff_param->identifier() : non_differrentiable_identifier;
    }
    [[nodiscard]] void update_by_buffer(Stream &stream, float4 new_value){
        LUISA_INFO("Constant::update_by_buffer {}", _constant_slot);
        pipeline().update_constant(stream, _constant_slot, new_value);
    }
};

luisa::unique_ptr<Texture::Instance> ConstantTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::optional<Differentiation::ConstantParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation()->parameter(_v, _channels, range()));
    }
    return luisa::make_unique<ConstantTextureInstance>(pipeline, this, command_buffer, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantTexture)
