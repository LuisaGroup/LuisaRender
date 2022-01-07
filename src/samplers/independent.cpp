//
// Created by Mike on 2022/1/7.
//

#include <luisa-compute.h>
#include <scene/pipeline.h>
#include <scene/sampler.h>

namespace luisa::render {

using namespace luisa::compute;

class IndependentSampler;

class IndependentSamplerInstance : public Sampler::Instance {

private:
    Device &_device;
    Image<uint> _states;
    luisa::optional<Var<uint>> _state;
    luisa::optional<Var<uint2>> _pixel;
    uint _seed;
    Shader2D<Image<uint>, uint> _make_sampler_state;

public:
    IndependentSamplerInstance(Device &device, const IndependentSampler *sampler, uint _seed) noexcept;
    void reset(Stream &stream, uint2 resolution, uint spp) noexcept override;
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
    [[nodiscard]] luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline) const noexcept override {
        return luisa::make_unique<IndependentSamplerInstance>(pipeline.device(), this, _seed);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "independent"; }
};

IndependentSamplerInstance::IndependentSamplerInstance(Device &device, const IndependentSampler *sampler, uint seed) noexcept
    : Sampler::Instance{sampler}, _device{device}, _seed{seed} {
    Kernel2D kernel = [](ImageUInt states, UInt seed) noexcept {
        auto tea = [](UInt v0, UInt v1) noexcept {
            auto s0 = def(0u);
            for (auto n = 0u; n < 4u; n++) {
                s0 += 0x9e3779b9u;
                v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
                v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
            }
            return v0;
        };
        auto p = dispatch_id().xy();
        auto state = tea(p.x, p.y);
        states.write(p, make_uint4(state));
    };
    _make_sampler_state = _device.compile(kernel);
}

void IndependentSamplerInstance::reset(Stream &stream, uint2 resolution, uint /* spp */) noexcept {
    if (!_states || any(_states.size() < resolution)) {
        _states = _device.create_image<uint>(PixelStorage::INT1, resolution);
    }
    stream << _make_sampler_state(_states, _seed).dispatch(resolution);
}

void IndependentSamplerInstance::start(Expr<uint2> pixel, Expr<uint> /* sample_index */) noexcept {
    load_state(pixel);
}

void IndependentSamplerInstance::save_state() noexcept {
    _states.write(*_pixel, make_uint4(*_state));
}

void IndependentSamplerInstance::load_state(Expr<uint2> pixel) noexcept {
    _pixel = luisa::nullopt;
    _pixel = def(pixel);
    _state = luisa::nullopt;
    _state = _states.read(pixel).x;
}

Float IndependentSamplerInstance::generate_1d() noexcept {
    auto lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
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
