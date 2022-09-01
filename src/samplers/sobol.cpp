//
// Created by Mike Smith on 2022/2/9.
//

#include <bit>

#include <dsl/sugar.h>
#include <util/u64.h>
#include <util/rng.h>
#include <util/sobolmatrices.h>
#include <base/sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class SobolSampler final : public Sampler {

public:
    SobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

using namespace luisa::compute;

class SobolSamplerInstance final : public Sampler::Instance {

private:
    uint _scale{};
    luisa::optional<UInt> _seed;
    luisa::optional<UInt> _dimension;
    luisa::optional<U64> _sobol_index;
    Buffer<uint> _sobol_matrices;
    luisa::unique_ptr<Constant<uint2>> _vdc_sobol_matrices;
    luisa::unique_ptr<Constant<uint2>> _vdc_sobol_matrices_inv;
    Buffer<uint4> _state_buffer;

private:
    [[nodiscard]] static auto _fast_owen_scramble(Expr<uint> seed, UInt v) noexcept {
        v = reverse(v);
        v ^= v * 0x3d20adeau;
        v += seed;
        v *= (seed >> 16u) | 1u;
        v ^= v * 0x05526c56u;
        v ^= v * 0x53a22864u;
        return reverse(v);
    }

    template<bool scramble>
    [[nodiscard]] auto _sobol_sample(U64 a, Expr<uint> dimension, Expr<uint> hash) const noexcept {
        auto v = def(0u);
        auto i = def(dimension * SobolMatrixSize);
        $while(a != 0u) {
            v = ite((a & 1u) != 0u, v ^ _sobol_matrices.read(i), v);
            a = a >> 1u;
            i = i + 1u;
        };
        if constexpr (scramble) { v = _fast_owen_scramble(hash, v); }
        return min(v * 0x1p-32f, one_minus_epsilon);
    }

    [[nodiscard]] auto _sobol_interval_to_index(uint m, UInt frame, Expr<uint2> p) const noexcept {
        if (m == 0u) { return U64{frame}; }
        U64 delta;
        auto c = def(0u);
        $while(frame != 0u) {
            auto vdc = _vdc_sobol_matrices->read(c);
            auto delta_bits = ite((frame & 1u) != 0u, delta.bits() ^ vdc, delta.bits());
            delta = U64{delta_bits};
            frame = frame >> 1u;
            c = c + 1u;
        };
        // flipped b
        auto m2 = m << 1u;
        auto index = U64{frame} << m2;
        auto b = delta ^ ((U64{p.x} << m) | p.y);
        auto d = def(0u);
        $while(b != 0u) {
            auto vdc_inv = _vdc_sobol_matrices_inv->read(d);
            auto index_bits = ite((b & 1u) != 0u, index.bits() ^ vdc_inv, index.bits());
            index = U64{index_bits};
            b = b >> 1u;
            d = d + 1u;
        };
        return index;
    }

public:
    explicit SobolSamplerInstance(
        const Pipeline &pipeline, CommandBuffer &command_buffer,
        const SobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s} {
        luisa::vector<uint4> vdc_sobol_matrices(SobolMatrixSize * VdCSobolMatrixSize);
        _sobol_matrices = pipeline.device().create_buffer<uint>(SobolMatrixSize * NSobolDimensions);
        command_buffer << _sobol_matrices.copy_from(SobolMatrices32)
                       << commit();
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        if (_state_buffer.size() < state_count) {
            _state_buffer = pipeline().device().create_buffer<uint4>(
                next_pow2(state_count));
        }
        _scale = next_pow2(std::max(resolution.x, resolution.y));
        auto m = std::bit_width(_scale);
        std::array<uint2, SobolMatrixSize> vdc_sobol_matrices;
        std::array<uint2, SobolMatrixSize> vdc_sobol_matrices_inv;
        for (auto i = 0u; i < SobolMatrixSize; i++) {
            vdc_sobol_matrices[i] = u64_to_uint2(VdCSobolMatrices[m - 1u][i]);
            vdc_sobol_matrices_inv[i] = u64_to_uint2(VdCSobolMatricesInv[m - 1u][i]);
        }
        _vdc_sobol_matrices = luisa::make_unique<Constant<uint2>>(vdc_sobol_matrices);
        _vdc_sobol_matrices_inv = luisa::make_unique<Constant<uint2>>(vdc_sobol_matrices_inv);
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        _dimension.emplace(2u);
        _sobol_index.emplace(_sobol_interval_to_index(
            std::bit_width(_scale), sample_index, pixel));
        _seed.emplace(xxhash32(make_uint4(
            pixel.x, pixel.y, sample_index, node()->seed())));
    }
    void save_state(Expr<uint> state_id) noexcept override {
        auto state = make_uint4(_sobol_index->bits(), *_dimension, *_seed);
        _state_buffer.write(state_id, state);
    }
    void load_state(Expr<uint> state_id) noexcept override {
        auto state = _state_buffer.read(state_id);
        _sobol_index.emplace(state.xy());
        _dimension.emplace(state.z);
        _seed.emplace(state.w);
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        auto u = _sobol_sample<true>(*_sobol_index, *_dimension, *_seed);
        *_dimension = max((*_dimension + 1u) % NSobolDimensions, 2u);
        return u;
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        auto ux = generate_1d();
        auto uy = generate_1d();
        return make_float2(ux, uy);
    }
    [[nodiscard]] Float2 generate_pixel_2d() noexcept override {
        auto ux = _sobol_sample<false>(*_sobol_index, 0u, *_seed);
        auto uy = _sobol_sample<false>(*_sobol_index, 1u, *_seed);
        return make_float2(ux, uy);
    }
};

luisa::unique_ptr<Sampler::Instance> SobolSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<SobolSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SobolSampler)
