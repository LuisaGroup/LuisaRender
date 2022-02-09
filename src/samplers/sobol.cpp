//
// Created by Mike Smith on 2022/2/9.
//

#include <dsl/sugar.h>
#include <util/sampling.h>
#include <util/sobolmatrices.h>
#include <base/sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class SobolSampler final : public Sampler {

private:
    uint _seed;

public:
    SobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc},
          _seed{desc->property_uint_or_default("seed", 19980810u)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto seed() const noexcept { return _seed; }
};

using namespace luisa::compute;

class SobolSamplerInstance final : public Sampler::Instance {

public:
    static constexpr auto max_dimension = 1024u;

private:
    uint _spp{};
    uint _width{};
    luisa::optional<UInt> _dimension{};
    luisa::optional<UInt> _pixel_index{};
    luisa::optional<UInt> _sample_index{};
    luisa::unique_ptr<Constant<uint>> _sobol_matrices;
    Buffer<uint2> _state_buffer;

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
    explicit SobolSamplerInstance(
        const Pipeline &pipeline, CommandBuffer &command_buffer,
        const SobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s} {
        std::array<uint, SobolMatrixSize * 2u> sobol_matrices{};
        std::memcpy(sobol_matrices.data(), SobolMatrices32, luisa::span{sobol_matrices}.size_bytes());
        _sobol_matrices = luisa::make_unique<Constant<uint>>(sobol_matrices);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        _spp = spp;
        auto pixel_count = resolution.x * resolution.y;
        if (_state_buffer.size() < pixel_count) {
            _state_buffer = pipeline().device().create_buffer<uint2>(
                next_pow2(pixel_count));
        }
        _width = resolution.x;
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        _dimension.emplace(0u);
        _pixel_index.emplace(pixel.y * _width + pixel.x);
        _sample_index.emplace(sample_index);
    }
    void save_state() noexcept override {
        auto state = make_uint2(*_sample_index, *_dimension);
        _state_buffer.write(*_pixel_index, state);
    }
    void load_state(Expr<uint2> pixel) noexcept override {
        _pixel_index.emplace(pixel.y * _width + pixel.x);
        auto state = _state_buffer.read(*_pixel_index);
        _sample_index.emplace(state.x);
        _dimension.emplace(state.y);
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        auto hash = xxhash32(make_uint3(*_pixel_index, *_sample_index, *_dimension));
        auto u = _sobol_sample(*_sample_index, 0u, hash);
        *_dimension += 1u;
        return u;
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        auto hash = xxhash32(make_uint3(*_pixel_index, *_sample_index, *_dimension));
        auto ux = _sobol_sample(*_sample_index, 0u, hash);
        auto uy = _sobol_sample(*_sample_index, 1u, (hash << 16u) | (hash >> 16u));
        *_dimension += 2u;
        return make_float2(ux, uy);
    }
};

luisa::unique_ptr<Sampler::Instance> SobolSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<SobolSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SobolSampler)
