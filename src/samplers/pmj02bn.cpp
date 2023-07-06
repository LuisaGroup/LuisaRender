//
// Created by Mike Smith on 2022/9/1.
//

#include <bit>

#include <dsl/sugar.h>
#include <util/rng.h>
#include <util/bluenoise.h>
#include <util/pmj02tables.h>
#include <base/sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class PMJ02BNSampler final : public Sampler {

public:
    PMJ02BNSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

using namespace luisa::compute;

class PMJ02BNSamplerInstance final : public Sampler::Instance {

private:
    uint _blue_noise_texture_id{0u};
    uint _sample_table_buffer_id{0u};
    uint _spp{0u};
    uint _w{0u};
    uint _pixel_tile_size{0u};
    Buffer<float2> _pixel_samples;
    Buffer<uint4> _state_buffer;
    luisa::optional<UInt2> _pixel;
    luisa::optional<UInt> _sample_index;
    luisa::optional<UInt> _dimension;

private:
    [[nodiscard]] static auto _blue_noise(Expr<uint> tex_index, Expr<uint2> p,
                                          Expr<BindlessArray> array, Expr<uint> bn_texture_id) noexcept {
        auto uvw = make_uint3(p.yx() % BlueNoiseResolution,
                              tex_index % NumBlueNoiseTextures);
        return array.tex3d(bn_texture_id).read(uvw).x;
    }

    [[nodiscard]] static auto _pmj02bn_sample(Expr<uint> set_id, Expr<uint> sample_id, Expr<BindlessArray> array, Expr<uint> buffer_id) noexcept {
        auto set_index = set_id % nPMJ02bnSets;
        auto i = set_index * nPMJ02bnSamples + sample_id;
        auto sample = array.buffer<uint2>(buffer_id).read(i);
        return make_float2(sample) * 0x1p-32f;
    }

    [[nodiscard]] static auto _permutation_element(Expr<uint> i, Expr<uint> l, Expr<uint> w, Expr<uint> p) noexcept {
        static Callable impl = [](UInt i, UInt w, UInt l, UInt p) noexcept {
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
        return impl(i, w, l, p);
    }

public:
    PMJ02BNSamplerInstance(Pipeline &pipeline, CommandBuffer &cb,
                           const PMJ02BNSampler *sampler) noexcept
        : Sampler::Instance{pipeline, sampler} {
        auto k = BlueNoiseResolution;
        auto n = NumBlueNoiseTextures;
        auto blue_noise_texture = pipeline.create<Volume<float>>(
            PixelStorage::SHORT1, make_uint3(k, k, n));
        _blue_noise_texture_id = pipeline.register_bindless(
            *blue_noise_texture, TextureSampler::point_repeat());
        auto sample_table_buffer = pipeline.create<Buffer<uint2>>(
            nPMJ02bnSets * nPMJ02bnSamples);
        _sample_table_buffer_id = pipeline.register_bindless(
            *sample_table_buffer);
        cb << blue_noise_texture->copy_from(BlueNoiseTextures)
           << sample_table_buffer->copy_from(PMJ02bnSamples);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution,
               uint state_count, uint spp) noexcept override {
        static constexpr auto log2 = [](auto x) noexcept { return std::bit_width(x) - 1u; };
        static constexpr auto log4 = [](auto x) noexcept { return log2(x) / 2u; };
        static constexpr auto is_pow4 = [](auto x) noexcept { return x == (1u << 2u * log4(x)); };
        static constexpr auto next_pow4 = [](auto x) noexcept { return is_pow4(x) ? x : 1u << (2u * (log4(x) + 1u)); };
        LUISA_ASSERT(spp <= nPMJ02bnSamples,
                     "PMJ02BNSampler only supports up to "
                     "{} samples per pixel ({} requested).",
                     nPMJ02bnSamples, spp);
        if (!is_pow4(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "PMJ02BNSampler results are best "
                "with power-of-4 samples per pixel.");
        }
        _spp = spp;
        _w = _spp - 1u;
        _w |= _w >> 1u;
        _w |= _w >> 2u;
        _w |= _w >> 4u;
        _w |= _w >> 8u;
        _w |= _w >> 16u;
        _pixel_tile_size = 1u << (log4(nPMJ02bnSamples) - log4(next_pow4(spp)));
        auto pixel_sample_count = _pixel_tile_size * _pixel_tile_size * spp;
        if (!_pixel_samples||_pixel_samples.size() < pixel_sample_count) {
            _pixel_samples = pipeline().device().create_buffer<float2>(
                next_pow2(pixel_sample_count));
        }
        if (!_state_buffer||_state_buffer.size() < state_count) {
            _state_buffer = pipeline().device().create_buffer<uint4>(
                next_pow2(state_count));
        }
        luisa::vector<float2> pixel_samples(pixel_sample_count, make_float2(0.f));
        luisa::vector<uint> stored_counts(_pixel_tile_size * _pixel_tile_size, 0u);
        auto pmj02bn_sample = [](auto set_id, auto sample_id) noexcept {
            auto sample = PMJ02bnSamples[set_id % nPMJ02bnSets][sample_id % nPMJ02bnSamples];
            return make_float2(static_cast<float>(sample[0] * 0x1p-32),
                               static_cast<float>(sample[1] * 0x1p-32));
        };
        for (auto i = 0u; i < nPMJ02bnSamples; i++) {
            auto p = pmj02bn_sample(0, i) * static_cast<float>(_pixel_tile_size);
            auto pixel_offset = static_cast<uint>(p.y) * _pixel_tile_size +
                                static_cast<uint>(p.x);
            if (auto count = stored_counts[pixel_offset]; count == spp) {
                LUISA_ASSERT(!is_pow4(spp),
                             "Invalid pixel sorting state "
                             "(index = {}, count = {}).",
                             i, count);
                continue;
            }
            auto sample_offset = pixel_offset * spp + stored_counts[pixel_offset];
            LUISA_ASSERT(all(pixel_samples[sample_offset] == make_float2(0.f)),
                         "Invalid pixel sorting state (index = {}).", i);
            pixel_samples[sample_offset] = fract(p);
            stored_counts[pixel_offset]++;
        }
        for (auto c : stored_counts) {
            LUISA_ASSERT(c == spp, "Invalid pixel sorting state.");
        }
        command_buffer << _pixel_samples.view(0u, pixel_sample_count)
                              .copy_from(pixel_samples.data())
                       << commit();
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        _pixel.emplace(pixel);
        _dimension.emplace(2u);
        _sample_index.emplace(sample_index);
    }
    void save_state(Expr<uint> state_id) noexcept override {
        _state_buffer->write(
            state_id, make_uint4(*_pixel, *_sample_index, *_dimension));
    }
    void load_state(Expr<uint> state_id) noexcept override {
        auto state = _state_buffer->read(state_id);
        _pixel.emplace(state.xy());
        _sample_index.emplace(state.z);
        _dimension.emplace(state.w);
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        static Callable impl = [](UInt2 pixel, UInt dimension, UInt seed, UInt sample_index,
                                  UInt spp, UInt w, BindlessVar array, UInt bn_tex_id) noexcept {
            auto hash = xxhash32(make_uint4(pixel, dimension, seed));
            auto index = _permutation_element(sample_index, spp, w, hash);
            auto delta = _blue_noise(dimension, pixel, array, bn_tex_id);
            auto u = (cast<float>(index) + delta) * cast<float>(1.f / spp);
            return clamp(u, 0.f, one_minus_epsilon);
        };
        auto u = impl(*_pixel, *_dimension, node()->seed(), *_sample_index,
                      _spp, _w, pipeline().bindless_array(), _blue_noise_texture_id);
        *_dimension += 1u;
        return u;
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        static Callable impl = [](UInt2 pixel, UInt dimension, UInt seed, UInt sample_index,
                                  UInt spp, UInt w, BindlessVar array, UInt bn_tex_id, UInt table) noexcept {
            auto index = sample_index;
            auto pmj_instance = dimension / 2u;
            $if(pmj_instance >= nPMJ02bnSets) {
                auto hash = xxhash32(make_uint4(pixel, dimension, seed));
                index = _permutation_element(sample_index, spp, w, hash);
            };
            auto u = _pmj02bn_sample(pmj_instance, index, array, table) +
                     make_float2(_blue_noise(dimension, pixel, array, bn_tex_id),
                                 _blue_noise(dimension + 1u, pixel, array, bn_tex_id));
            return fract(u);
        };
        auto u = impl(*_pixel, *_dimension, node()->seed(), *_sample_index, _spp, _w,
                      pipeline().bindless_array(), _blue_noise_texture_id, _sample_table_buffer_id);
        *_dimension += 2u;
        return u;
    }
    [[nodiscard]] Float2 generate_pixel_2d() noexcept override {
        auto p = *_pixel % _pixel_tile_size;
        auto offset = (p.x + p.y * _pixel_tile_size) * _spp;
        return _pixel_samples->read(offset + *_sample_index);
    }
};

luisa::unique_ptr<Sampler::Instance> PMJ02BNSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PMJ02BNSamplerInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PMJ02BNSampler)
