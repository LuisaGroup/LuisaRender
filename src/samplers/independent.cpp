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
    uint2 _resolution;
    Buffer<uint> _states;
    luisa::optional<Var<uint>> _state;
    luisa::optional<Var<uint>> _pixel_id;

public:
    IndependentSamplerInstance(const Pipeline &pipeline, const IndependentSampler *sampler) noexcept;
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint spp) noexcept override;
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override;
    void save_state() noexcept override;
    void load_state(Expr<uint2> pixel) noexcept override;
    Float generate_1d() noexcept override;
    Float2 generate_2d() noexcept override;
};

class IndependentSampler final : public Sampler {

private:
    uint _seed;

public:
    IndependentSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc}, _seed{desc->property_uint_or_default("seed", 19980810u)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<IndependentSamplerInstance>(pipeline, this);
    }
    [[nodiscard]] auto seed() const noexcept { return _seed; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

IndependentSamplerInstance::IndependentSamplerInstance(
    const Pipeline &pipeline, const IndependentSampler *sampler) noexcept
    : Sampler::Instance{pipeline, sampler} {}

void IndependentSamplerInstance::reset(CommandBuffer &command_buffer, uint2 resolution, uint /* spp */) noexcept {
    _resolution = resolution;
    auto pixel_count = _resolution.x * _resolution.y;
    if (!_states || pixel_count > _states.size()) {
        _states = pipeline().device().create_buffer<uint>(next_pow2(pixel_count));
    }
}

void IndependentSamplerInstance::start(Expr<uint2> pixel, Expr<uint> index) noexcept {
    auto seed = static_cast<const IndependentSampler *>(node())->seed();
    _pixel_id.emplace(pixel.y * _resolution.x + pixel.x);
    _state.emplace(xxhash32(make_uint3(seed, index, *_pixel_id)));
}

void IndependentSamplerInstance::save_state() noexcept {
    _states.write(*_pixel_id, *_state);
}

void IndependentSamplerInstance::load_state(Expr<uint2> pixel) noexcept {
    _pixel_id.emplace(pixel.y * _resolution.x + pixel.x);
    _state.emplace(_states.read(pixel.y * _resolution.x + pixel.x));
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
