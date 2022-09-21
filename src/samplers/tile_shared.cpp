//
// Created by Mike Smith on 2022/9/20.
//

#include <base/sampler.h>
#include <base/scene.h>
#include <util/rng.h>

namespace luisa::render {

class TileSharedSampler final : public Sampler {

private:
    Sampler *_base;
    uint2 _tile_size;
    bool _jitter;

public:
    TileSharedSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc},
          _base{scene->load_sampler(desc->property_node("base"))},
          _tile_size{desc->property_uint2_or_default(
              "tile_size", lazy_construct([desc] {
                  auto s = desc->property_uint_or_default("tile_size", 16u);
                  return make_uint2(s);
              }))},
          _jitter{desc->property_bool_or_default("jitter", false)} {}
    [[nodiscard]] auto tile_size() const noexcept { return _tile_size; }
    [[nodiscard]] auto jitter() const noexcept { return _jitter; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class TileSharedSamplerInstance final : public Sampler::Instance {

private:
    luisa::unique_ptr<Sampler::Instance> _base;
    uint2 _tile_size;
    uint2 _resolution;

public:
    TileSharedSamplerInstance(const Pipeline &pipeline, const TileSharedSampler *sampler,
                              luisa::unique_ptr<Sampler::Instance> base) noexcept
        : Sampler::Instance{pipeline, sampler}, _base{std::move(base)} {}
    void reset(CommandBuffer &command_buffer, uint2 resolution,
               uint state_count, uint spp) noexcept override {
        _tile_size = luisa::min(resolution, node<TileSharedSampler>()->tile_size());
        _resolution = resolution;
        auto tile_count = (resolution + _tile_size - 1u) / _tile_size;
        _base->reset(command_buffer, tile_count, state_count, spp);
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        auto p = def(pixel);
        if (node<TileSharedSampler>()->jitter()) {
            auto offset = xxhash32(sample_index);
            auto o = make_float2(make_uint2(offset >> 16u, offset & 0xffffu)) * 0x1p-16f;
            p += make_uint2(o * make_float2(_resolution)) % _resolution;
        }
        auto tile = p / _tile_size;
        _base->start(tile, sample_index);
    }
    void save_state(Expr<uint> state_id) noexcept override {
        _base->save_state(state_id);
    }
    void load_state(Expr<uint> state_id) noexcept override {
        _base->load_state(state_id);
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        return _base->generate_1d();
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        return _base->generate_2d();
    }
    [[nodiscard]] Float2 generate_pixel_2d() noexcept override {
        return _base->generate_pixel_2d();
    }
};

luisa::unique_ptr<Sampler::Instance> TileSharedSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto base = _base->build(pipeline, command_buffer);
    return luisa::make_unique<TileSharedSamplerInstance>(
        pipeline, this, std::move(base));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::TileSharedSampler)
