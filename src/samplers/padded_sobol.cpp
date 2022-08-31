//
// Created by Mike Smith on 2022/2/9.
//

#include <dsl/sugar.h>
#include <util/rng.h>
#include <util/sobolmatrices.h>
#include <base/sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class PaddedSobolSampler final : public Sampler {

private:
    uint _seed;

public:
    PaddedSobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc},
          _seed{desc->property_uint_or_default("seed", 19980810u)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto seed() const noexcept { return _seed; }
};

using namespace luisa::compute;

class PaddedSobolSamplerInstance final : public Sampler::Instance {

private:
    uint _width{};
    luisa::optional<UInt> _seed;
    luisa::optional<UInt> _dimension;
    luisa::optional<UInt> _sample_index;
    luisa::unique_ptr<Constant<uint>> _sobol_matrices;
    Buffer<uint3> _state_buffer;

private:
    [[nodiscard]] static auto _fast_owen_scramble(UInt seed, UInt v) noexcept {
        v = reverse(v);
        v ^= v * 0x3d20adeau;
        v += seed;
        v *= (seed >> 16u) | 1u;
        v ^= v * 0x05526c56u;
        v ^= v * 0x53a22864u;
        return reverse(v);
    }

    [[nodiscard]] auto _sobol_sample(UInt a, uint dimension, Expr<uint> hash) const noexcept {
        auto v = def(0u);
        auto i = def(dimension * SobolMatrixSize);
        $while(a != 0u) {
            v = ite((a & 1u) != 0u, v ^ _sobol_matrices->read(i), v);
            a = a >> 1u;
            i += 1u;
        };
        v = _fast_owen_scramble(hash, v);
        return min(v * 0x1p-32f, one_minus_epsilon);
    }

public:
    explicit PaddedSobolSamplerInstance(
        const Pipeline &pipeline, CommandBuffer &command_buffer,
        const PaddedSobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s} {
        std::array<uint, SobolMatrixSize * 2u> sobol_matrices{};
        std::memcpy(sobol_matrices.data(), SobolMatrices32, luisa::span{sobol_matrices}.size_bytes());
        _sobol_matrices = luisa::make_unique<Constant<uint>>(sobol_matrices);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        if (_state_buffer.size() < state_count) {
            _state_buffer = pipeline().device().create_buffer<uint3>(
                next_pow2(state_count));
        }
        _width = resolution.x;
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        _dimension.emplace(0u);
        _sample_index.emplace(sample_index);
        _seed.emplace(xxhash32(make_uint4(
            pixel.x, pixel.y, sample_index,
            node<PaddedSobolSampler>()->seed())));
    }
    void save_state(Expr<uint> state_id) noexcept override {
        auto state = make_uint3(*_seed, *_sample_index, *_dimension);
        _state_buffer.write(state_id, state);
    }
    void load_state(Expr<uint> state_id) noexcept override {
        auto state = _state_buffer.read(state_id);
        _seed.emplace(state.x);
        _sample_index.emplace(state.y);
        _dimension.emplace(state.z);
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        auto hash = xxhash32(make_uint2(*_dimension, *_seed));
        auto u = _sobol_sample(*_sample_index, 0u, hash);
        *_dimension += 1u;
        return u;
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        auto hx = xxhash32(make_uint2(*_dimension, *_seed));
        auto hy = xxhash32(make_uint2(*_dimension + 1u, *_seed));
        auto ux = _sobol_sample(*_sample_index, 0u, hx);
        auto uy = _sobol_sample(*_sample_index, 1u, hy);
        *_dimension += 2u;
        return make_float2(ux, uy);
    }
};

luisa::unique_ptr<Sampler::Instance> PaddedSobolSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PaddedSobolSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PaddedSobolSampler)
