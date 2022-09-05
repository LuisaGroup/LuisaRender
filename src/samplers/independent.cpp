//
// Created by Mike on 2022/1/7.
//

#include <luisa-compute.h>

#include <util/rng.h>
#include <base/pipeline.h>
#include <base/sampler.h>

namespace luisa::render {

using namespace luisa::compute;

class IndependentSampler;

class IndependentSamplerInstance final : public Sampler::Instance {

private:
    Buffer<uint> _states;
    luisa::optional<Var<uint>> _state;

public:
    IndependentSamplerInstance(const Pipeline &pipeline, const IndependentSampler *sampler) noexcept;
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept override;
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override;
    void save_state(Expr<uint> state_id) noexcept override;
    void load_state(Expr<uint> state_id) noexcept override;
    Float generate_1d() noexcept override;
    Float2 generate_2d() noexcept override;
};

class IndependentSampler final : public Sampler {

public:
    IndependentSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<IndependentSamplerInstance>(pipeline, this);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

IndependentSamplerInstance::IndependentSamplerInstance(
    const Pipeline &pipeline, const IndependentSampler *sampler) noexcept
    : Sampler::Instance{pipeline, sampler} {}

void IndependentSamplerInstance::reset(
    CommandBuffer &command_buffer, uint2, uint state_count, uint) noexcept {
    if (!_states || state_count > _states.size()) {
        _states = pipeline().device().create_buffer<uint>(
            next_pow2(state_count));
    }
}

void IndependentSamplerInstance::start(Expr<uint2> pixel, Expr<uint> index) noexcept {
    _state.emplace(xxhash32(make_uint3(node()->seed(), index, (pixel.x << 16u) | pixel.y)));
}

void IndependentSamplerInstance::save_state(Expr<uint> state_id) noexcept {
    _states.write(state_id, *_state);
}

void IndependentSamplerInstance::load_state(Expr<uint> state_id) noexcept {
    _state.emplace(_states.read(state_id));
}

Float IndependentSamplerInstance::generate_1d() noexcept {
    auto lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };
    return lcg(*_state);
}

Float2 IndependentSamplerInstance::generate_2d() noexcept {
    auto ux = generate_1d();
    auto uy = generate_1d();
    return make_float2(ux, uy);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::IndependentSampler)
