//
// Created by Mike on 2022/1/7.
//

#include <luisa-compute.h>
#include <base/pipeline.h>
#include <base/sampler.h>

namespace luisa::render {

using namespace luisa::compute;

class IndependentSampler;

class IndependentSamplerInstance final : public Sampler::Instance {

private:
    Device &_device;
    uint2 _resolution;
    Buffer<uint> _states;
    luisa::optional<Var<uint>> _state;
    luisa::optional<Var<uint>> _pixel_id;
    uint _seed;
    Shader1D<Buffer<uint>, uint, uint> _make_sampler_state;

public:
    IndependentSamplerInstance(Device &device, const IndependentSampler *sampler, uint _seed) noexcept;
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
        return luisa::make_unique<IndependentSamplerInstance>(pipeline.device(), this, _seed);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

IndependentSamplerInstance::IndependentSamplerInstance(Device &device, const IndependentSampler *sampler, uint seed) noexcept
    : Sampler::Instance{sampler}, _device{device}, _seed{seed} {
    Kernel1D kernel = [](BufferUInt states, UInt seed, UInt width) noexcept {
        auto tea = [](UInt s0, UInt v0, UInt v1) noexcept {
            for (auto n = 0u; n < 4u; n++) {
                s0 += 0x9e3779b9u;
                v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
                v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
            }
            return v0;
        };
        auto x = dispatch_x() % width;
        auto y = dispatch_x() / width;
        auto state = tea(seed, x, y);
        states.write(dispatch_x(), state);
    };
    _make_sampler_state = _device.compile(kernel);
}

void IndependentSamplerInstance::reset(CommandBuffer &command_buffer, uint2 resolution, uint /* spp */) noexcept {
    _resolution = resolution;
    auto pixel_count = _resolution.x * _resolution.y;
    if (!_states || pixel_count > _states.size()) {
        _states = _device.create_buffer<uint>(next_pow2(pixel_count));
    }
    command_buffer << _make_sampler_state(_states, _seed, _resolution.x).dispatch(pixel_count);
}

void IndependentSamplerInstance::start(Expr<uint2> pixel, Expr<uint> /* sample_index */) noexcept {
    load_state(pixel);
}

void IndependentSamplerInstance::save_state() noexcept {
    _states.write(*_pixel_id, *_state);
}

void IndependentSamplerInstance::load_state(Expr<uint2> pixel) noexcept {
    _pixel_id = luisa::nullopt;
    _pixel_id = pixel.y * _resolution.x + pixel.x;;
    _state = luisa::nullopt;
    _state = _states.read(pixel.y * _resolution.x + pixel.x);
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
