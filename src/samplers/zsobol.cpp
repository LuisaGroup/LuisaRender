//
// Created by Mike Smith on 2022/2/9.
//

#include <dsl/sugar.h>
#include <util/u64.h>
#include <util/sobolmatrices.h>
#include <base/sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class ZSobolSampler final : public Sampler {

public:
    ZSobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

using namespace luisa::compute;

class ZSobolSamplerInstance final : public Sampler::Instance {

public:
    static constexpr auto max_dimension = 1024u;

private:
    uint _log2_spp{};
    luisa::optional<UInt> _dimension{};
    luisa::optional<U64> _morton_index{};
    luisa::unique_ptr<Constant<uint2>> _sample_hash;
    Buffer<uint> _sobol_matrices;
    Buffer<uint3> _state_buffer;
    luisa::unique_ptr<Callable<uint2(uint2, uint)>> _get_sample_index_impl;

private:
    [[nodiscard]] auto _get_sample_index() const noexcept {
        return U64{(*_get_sample_index_impl)(_morton_index->bits(), *_dimension)};
    }

    [[nodiscard]] static auto _fast_owen_scramble(UInt seed, UInt v) noexcept {
        v = reverse(v);
        v ^= v * 0x3d20adeau;
        v += seed;
        v *= (seed >> 16u) | 1u;
        v ^= v * 0x05526c56u;
        v ^= v * 0x53a22864u;
        return reverse(v);
    }

    [[nodiscard]] auto _sobol_sample(U64 a, uint dimension, Expr<uint> hash) const noexcept {
        static Callable impl = [](UInt2 a_bits, UInt dimension, UInt hash, BufferVar<uint> sobol_matrices) noexcept {
            auto a = U64{a_bits};
            auto v = def(0u);
            auto i = def(dimension * SobolMatrixSize);
            $while(a != 0u) {
                v = ite((a & 1u) != 0u, v ^ sobol_matrices.read(i), v);
                a = a >> 1u;
                i += 1u;
            };
            v = _fast_owen_scramble(hash, v);
            return min(v * 0x1p-32f, one_minus_epsilon);
        };
        return impl(a.bits(), dimension, hash, _sobol_matrices.view());
    }

public:
    explicit ZSobolSamplerInstance(const Pipeline &pipeline, CommandBuffer &command_buffer, const ZSobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s} {
        std::array<uint2, max_dimension> sample_hash{};
        for (auto i = 0u; i < max_dimension; i++) {
            auto u = (static_cast<uint64_t>(s->seed()) << 32u) | i;
            auto hash = hash_value(u);
            sample_hash[i] = luisa::make_uint2(
                static_cast<uint>(hash & ~0u),
                static_cast<uint>(hash >> 32u));
        }
        std::array<uint, SobolMatrixSize * 2u> sobol_matrices{};
        _sobol_matrices = pipeline.device().create_buffer<uint>(SobolMatrixSize * 2u);
        command_buffer << _sobol_matrices.copy_from(SobolMatrices32);
        _sample_hash = luisa::make_unique<Constant<uint2>>(sample_hash);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        if (!_state_buffer || _state_buffer.size() < state_count) {
            _state_buffer = pipeline().device().create_buffer<uint3>(
                next_pow2(state_count));
        }
        static constexpr auto log2 = [](auto x) noexcept {
            return std::bit_width(next_pow2(x)) - 1u;
        };
        _log2_spp = log2(spp);
        _get_sample_index_impl = luisa::make_unique<Callable<uint2(uint2, uint)>>(
            [this, resolution](UInt2 morton, UInt dimension) noexcept {
                static Constant<uint4> permutations{std::array{
                    make_uint4(0, 1, 2, 3), make_uint4(0, 1, 3, 2), make_uint4(0, 2, 1, 3), make_uint4(0, 2, 3, 1),
                    make_uint4(0, 3, 2, 1), make_uint4(0, 3, 1, 2), make_uint4(1, 0, 2, 3), make_uint4(1, 0, 3, 2),
                    make_uint4(1, 2, 0, 3), make_uint4(1, 2, 3, 0), make_uint4(1, 3, 2, 0), make_uint4(1, 3, 0, 2),
                    make_uint4(2, 1, 0, 3), make_uint4(2, 1, 3, 0), make_uint4(2, 0, 1, 3), make_uint4(2, 0, 3, 1),
                    make_uint4(2, 3, 0, 1), make_uint4(2, 3, 1, 0), make_uint4(3, 1, 2, 0), make_uint4(3, 1, 0, 2),
                    make_uint4(3, 2, 1, 0), make_uint4(3, 2, 0, 1), make_uint4(3, 0, 2, 1), make_uint4(3, 0, 1, 2)}};
                static constexpr auto mix_bits = [](U64 v) noexcept {
                    v = v ^ (v >> 31u);
                    v = v * U64{0x7fb5d329728ea185ull};
                    v = v ^ (v >> 27u);
                    v = v * U64{0x81dadef4bc2dd44dull};
                    v = v ^ (v.hi() >> 1u);
                    return v;
                };
                U64 sample_index;
                auto log4_spp = (_log2_spp + 1u) / 2u;
                auto pow2_samples = static_cast<bool>(_log2_spp & 1u);
                auto last_digit = pow2_samples ? 1 : 0;
                auto morton_index = U64{morton};
                auto num_base4_digits = log2(std::max(resolution.x, resolution.y)) + log4_spp;
                for (auto i = static_cast<int>(num_base4_digits) - 1; i >= last_digit; i--) {
                    auto digit_shift = 2u * i - (pow2_samples ? 1u : 0u);
                    auto digit = (morton_index >> digit_shift) & 3u;
                    auto higher_digits = morton_index >> (digit_shift + 2u);
                    auto p = (mix_bits(higher_digits ^ (dimension * 0x55555555u)) >> 24u) % 24u;
                    U64 perm_digit{permutations.read(p)[digit]};
                    sample_index = sample_index | (perm_digit << digit_shift);
                }
                if (pow2_samples) {
                    auto digit = (morton_index & 1u) ^
                                 (mix_bits((morton_index >> 1u) ^ (dimension * 0x55555555u)) & 1u);
                    sample_index = sample_index | digit;
                }
                return sample_index.bits();
            });
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        static Callable encode_morton = [](UInt x, UInt y) noexcept {
            static constexpr auto left_shift2 = [](auto x_in) noexcept {
                U64 x{x_in};
                x = (x ^ (x << 16u)) & U64{0x0000ffff0000ffffull};
                x = (x ^ (x << 8u)) & U64{0x00ff00ff00ff00ffull};
                x = (x ^ (x << 4u)) & U64{0x0f0f0f0f0f0f0f0full};
                x = (x ^ (x << 2u)) & U64{0x3333333333333333ull};
                x = (x ^ (x << 1u)) & U64{0x5555555555555555ull};
                return x;
            };
            return ((left_shift2(y) << 1u) | left_shift2(x)).bits();
        };
        _dimension = luisa::nullopt;
        _morton_index = luisa::nullopt;
        _dimension = def(0u);
        _morton_index = (U64{encode_morton(pixel.x, pixel.y)} << _log2_spp) | sample_index;
    }
    void save_state(Expr<uint> state_id) noexcept override {
        auto state = make_uint3(_morton_index->bits(), *_dimension);
        _state_buffer->write(state_id, state);
    }
    void load_state(Expr<uint> state_id) noexcept override {
        _dimension = luisa::nullopt;
        _morton_index = luisa::nullopt;
        auto state = _state_buffer->read(state_id);
        _morton_index = U64{state.xy()};
        _dimension = state.z;
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        auto sample_index = _get_sample_index();
        auto sample_hash = _sample_hash->read(*_dimension).x;
        *_dimension = (*_dimension + 1u) % max_dimension;
        return _sobol_sample(sample_index, 0u, sample_hash);
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        auto sample_index = _get_sample_index();
        auto sample_hash = _sample_hash->read(*_dimension);
        *_dimension = (*_dimension + 2u) % max_dimension;
        auto ux = _sobol_sample(sample_index, 0u, sample_hash.x);
        auto uy = _sobol_sample(sample_index, 1u, sample_hash.y);
        return make_float2(ux, uy);
    }
};

luisa::unique_ptr<Sampler::Instance> ZSobolSampler::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ZSobolSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ZSobolSampler)