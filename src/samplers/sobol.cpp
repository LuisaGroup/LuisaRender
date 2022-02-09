//
// Created by Mike Smith on 2022/2/9.
//

#include <dsl/sugar.h>
#include <util/u64.h>
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
    uint _seed{};
    uint _log2_spp{};
    uint _num_base4_digits{};
    uint _width{};
    luisa::optional<UInt> _pixel_index{};
    luisa::optional<UInt> _dimension{};
    luisa::optional<U64> _morton_index{};
    luisa::unique_ptr<Constant<uint>> _sample_hash;
    Buffer<uint3> _state_buffer;

public:
    explicit SobolSamplerInstance(const Pipeline &pipeline, CommandBuffer &cb, const SobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s} {
        std::array<uint, max_dimension> sample_hash{};
        for (auto i = 0u; i < max_dimension; i++) {
            auto u = (static_cast<uint64_t>(s->seed()) << 32u) | i;
            sample_hash[i] = hash64(u);
        }
        _sample_hash = luisa::make_unique<Constant<uint>>(sample_hash);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        auto log2_uint = [](auto x) noexcept {
            auto n = 0u;
            for (;(1u << n) < x; n++) {}
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
        return 1.f;
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        return make_float2(1.f);
    }
};

luisa::unique_ptr<Sampler::Instance> SobolSampler::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<SobolSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render