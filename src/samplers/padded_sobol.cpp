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

public:
    PaddedSobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

using namespace luisa::compute;

class PaddedSobolSamplerInstance final : public Sampler::Instance {

private:
    luisa::optional<UInt2> _pixel;
    luisa::optional<UInt> _dimension;
    luisa::optional<UInt> _sample_index;
    Buffer<uint> _sobol_matrices;
    Buffer<uint4> _state_buffer;
    uint _spp{};

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

    [[nodiscard]] auto _sobol_sample(UInt a, Expr<uint> dimension, Expr<uint> hash) const noexcept {
        static Callable impl = [](UInt a, UInt dimension, UInt hash, BufferVar<uint> sobol_matrices) noexcept {
            auto v = def(0u);
            auto i = def(dimension * SobolMatrixSize);
            $while(a != 0u) {
                $if((a & 1u) != 0u) { v = v ^ sobol_matrices.read(i); };
                a >>= 1u;
                i += 1u;
            };
            v = _fast_owen_scramble(hash, v);
            return min(v * 0x1p-32f, one_minus_epsilon);
        };
        return impl(a, dimension, hash, _sobol_matrices.view());
    }

    [[nodiscard]] static auto _permutation_element(Expr<uint> i, uint l, Expr<uint> p) noexcept {
        auto w = l - 1u;
        w |= w >> 1u;
        w |= w >> 2u;
        w |= w >> 4u;
        w |= w >> 8u;
        w |= w >> 16u;
        static Callable impl = [](UInt i, UInt l, UInt w, UInt p) noexcept {
            $loop {
                i ^= p;
                i *= 0xe170893du;
                i ^= p >> 16u;
                i ^= (i & w) >> 4u;
                i ^= p >> 8u;
                i *= 0x0929eb3fu;
                i ^= p >> 23u;
                i ^= (i & w) >> 1u;
                i *= 1 | p >> 27u;
                i *= 0x6935fa69u;
                i ^= (i & w) >> 11u;
                i *= 0x74dcb303u;
                i ^= (i & w) >> 2u;
                i *= 0x9e501cc3u;
                i ^= (i & w) >> 2u;
                i *= 0xc860a3dfu;
                i &= w;
                i ^= i >> 5u;
                $if(i < l) { $break; };
            };
            return (i + p) % l;
        };
        return impl(i, l, w, p);
    }

public:
    explicit PaddedSobolSamplerInstance(
        const Pipeline &pipeline, CommandBuffer &command_buffer,
        const PaddedSobolSampler *s) noexcept
        : Sampler::Instance{pipeline, s},
          _sobol_matrices{pipeline.device().create_buffer<uint>(SobolMatrixSize * 2u)} {
        command_buffer << _sobol_matrices.copy_from(SobolMatrices32);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept override {
        if (next_pow2(spp) != spp) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        if (!_state_buffer || _state_buffer.size() < state_count) {
            _state_buffer = pipeline().device().create_buffer<uint4>(
                next_pow2(state_count));
        }
        _spp = spp;
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        _dimension.emplace(0u);
        _sample_index.emplace(sample_index);
        _pixel.emplace(pixel);
    }
    void save_state(Expr<uint> state_id) noexcept override {
        auto state = make_uint4(*_pixel, *_sample_index, *_dimension);
        _state_buffer->write(state_id, state);
    }
    void load_state(Expr<uint> state_id) noexcept override {
        auto state = _state_buffer->read(state_id);
        _pixel.emplace(state.xy());
        _sample_index.emplace(state.z);
        _dimension.emplace(state.w);
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        auto hash = xxhash32(make_uint4(*_pixel, *_sample_index ^ node()->seed(), *_dimension));
        auto index = _permutation_element(*_sample_index, _spp, hash);
        auto u = _sobol_sample(index, 0u, hash);
        *_dimension += 1u;
        return u;
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        auto hx = xxhash32(make_uint4(*_pixel, *_sample_index ^ node()->seed(), *_dimension));
        auto hy = xxhash32(make_uint4(*_pixel, *_sample_index ^ node()->seed(), *_dimension + 1u));
        auto index = _permutation_element(*_sample_index, _spp, hx);
        auto ux = _sobol_sample(index, 0u, hx);
        auto uy = _sobol_sample(index, 1u, hy);
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
