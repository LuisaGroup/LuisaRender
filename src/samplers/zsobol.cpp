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

private:
    uint _seed;

public:
    ZSobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc},
          _seed{desc->property_uint_or_default("seed", 19980810u)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto seed() const noexcept { return _seed; }
};

using namespace luisa::compute;

class ZSobolSamplerInstance final : public Sampler::Instance {

public:
    static constexpr auto max_dimension = 1024u;

private:
    uint _log2_spp{};
    uint _num_base4_digits{};
    uint _width{};
    luisa::optional<UInt> _pixel_index{};
    luisa::optional<UInt> _dimension{};
    luisa::optional<U64> _morton_index{};
    luisa::unique_ptr<Constant<uint2>> _sample_hash;
    luisa::unique_ptr<Constant<uint4>> _permutations;
    luisa::unique_ptr<Constant<uint>> _sobol_matrices;
    Buffer<uint3> _state_buffer;

private:
    [[nodiscard]] auto _get_sample_index() const noexcept {
        static constexpr auto mix_bits = [](U64 v) noexcept {
            v = v ^ (v >> 31u);
            v = v * U64{0x7fb5d329728ea185ull};
            v = v ^ (v >> 27u);
            v = v * U64{0x81dadef4bc2dd44dull};
            v = v ^ (v.hi() >> 1u);
            return v;
        };
        U64 sample_index;
        auto pow2_samples = static_cast<bool>(_log2_spp & 1u);
        auto last_digit = pow2_samples ? 1 : 0;
        for (auto i = static_cast<int>(_num_base4_digits) - 1; i >= last_digit; i--) {
            auto digit_shift = 2u * i - (pow2_samples ? 1u : 0u);
            auto digit = (*_morton_index >> digit_shift) & 3u;
            auto higher_digits = *_morton_index >> (digit_shift + 2u);
            auto p = (mix_bits(higher_digits ^ (*_dimension * 0x55555555u)) >> 24u) % 24u;
            U64 perm_digit{_permutations->read(p)[digit]};
            sample_index = sample_index | (perm_digit << digit_shift);
        }
        if (pow2_samples) {
            auto digit = (*_morton_index & 1u) ^
                         (mix_bits((*_morton_index >> 1u) ^ (*_dimension * 0x55555555u)) & 1u);
            sample_index = sample_index | digit;
        }
        return sample_index;
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
    explicit ZSobolSamplerInstance(const Pipeline &pipeline, const ZSobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s} {
        std::array<uint2, max_dimension> sample_hash{};
        for (auto i = 0u; i < max_dimension; i++) {
            auto u = (static_cast<uint64_t>(s->seed()) << 32u) | i;
            auto hash = hash64(u);
            sample_hash[i] = luisa::make_uint2(
                static_cast<uint>(hash & ~0u),
                static_cast<uint>(hash >> 32u));
        }
        std::array permutations{
            make_uint4(0, 1, 2, 3), make_uint4(0, 1, 3, 2), make_uint4(0, 2, 1, 3), make_uint4(0, 2, 3, 1),
            make_uint4(0, 3, 2, 1), make_uint4(0, 3, 1, 2), make_uint4(1, 0, 2, 3), make_uint4(1, 0, 3, 2),
            make_uint4(1, 2, 0, 3), make_uint4(1, 2, 3, 0), make_uint4(1, 3, 2, 0), make_uint4(1, 3, 0, 2),
            make_uint4(2, 1, 0, 3), make_uint4(2, 1, 3, 0), make_uint4(2, 0, 1, 3), make_uint4(2, 0, 3, 1),
            make_uint4(2, 3, 0, 1), make_uint4(2, 3, 1, 0), make_uint4(3, 1, 2, 0), make_uint4(3, 1, 0, 2),
            make_uint4(3, 2, 1, 0), make_uint4(3, 2, 0, 1), make_uint4(3, 0, 2, 1), make_uint4(3, 0, 1, 2)};
        std::array<uint, SobolMatrixSize * 2u> sobol_matrices{};
        std::memcpy(sobol_matrices.data(), SobolMatrices32, luisa::span{sobol_matrices}.size_bytes());
        _sample_hash = luisa::make_unique<Constant<uint2>>(sample_hash);
        _permutations = luisa::make_unique<Constant<uint4>>(permutations);
        _sobol_matrices = luisa::make_unique<Constant<uint>>(sobol_matrices);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        auto log2_uint = [](auto x) noexcept {
            auto n = 0u;
            for (; (1u << n) < x; n++) {}
            return n;
        };
        _log2_spp = log2_uint(spp);
        auto res = next_pow2(std::max(resolution.x, resolution.y));
        auto log4_spp = (_log2_spp + 1u) / 2u;
        _num_base4_digits = log2_uint(res) + log4_spp;
        auto pixel_count = resolution.x * resolution.y;
        if (_state_buffer.size() < pixel_count) {
            _state_buffer = pipeline().device().create_buffer<uint3>(
                next_pow2(pixel_count));
        }
        _width = resolution.x;
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        static constexpr auto left_shift2 = [](auto x_in) noexcept {
            U64 x{x_in};
            x = (x ^ (x << 16u)) & U64{0x0000ffff0000ffffull};
            x = (x ^ (x << 8u)) & U64{0x00ff00ff00ff00ffull};
            x = (x ^ (x << 4u)) & U64{0x0f0f0f0f0f0f0f0full};
            x = (x ^ (x << 2u)) & U64{0x3333333333333333ull};
            x = (x ^ (x << 1u)) & U64{0x5555555555555555ull};
            return x;
        };
        static constexpr auto encode_morton = [](auto x, auto y) noexcept -> U64 {
            return (left_shift2(y) << 1u) | left_shift2(x);
        };
        _dimension = luisa::nullopt;
        _pixel_index = luisa::nullopt;
        _morton_index = luisa::nullopt;
        _pixel_index = pixel.y * _width + pixel.x;
        _dimension = def(0u);
        _morton_index = (encode_morton(pixel.x, pixel.y) << _log2_spp) | sample_index;
    }
    void save_state() noexcept override {
        auto state = make_uint3(_morton_index->bits(), *_dimension);
        _state_buffer.write(*_pixel_index, state);
    }
    void load_state(Expr<uint2> pixel) noexcept override {
        _dimension = luisa::nullopt;
        _pixel_index = luisa::nullopt;
        _morton_index = luisa::nullopt;
        _pixel_index = pixel.y * _width + pixel.x;
        auto state = _state_buffer.read(*_pixel_index);
        _morton_index = U64{state.xy()};
        _dimension = state.z;
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        auto sample_index = _get_sample_index();
        auto sample_hash = _sample_hash->read(*_dimension).x;
        *_dimension += 1u;
        return _sobol_sample(sample_index, 0u, sample_hash);
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        auto sample_index = _get_sample_index();
        auto sample_hash = _sample_hash->read(*_dimension);
        *_dimension += 2u;
        auto ux = _sobol_sample(sample_index, 0u, sample_hash.x);
        auto uy = _sobol_sample(sample_index, 1u, sample_hash.y);
        return make_float2(ux, uy);
    }
};

luisa::unique_ptr<Sampler::Instance> ZSobolSampler::build(Pipeline &pipeline, CommandBuffer &) const noexcept {
    return luisa::make_unique<ZSobolSamplerInstance>(pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ZSobolSampler)
