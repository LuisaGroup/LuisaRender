//
// Created by Mike Smith on 2022/1/25.
//

#pragma once

#include <dsl/syntax.h>
#include <util/spectrum.h>
#include <util/imageio.h>
#include <base/scene_node.h>

namespace luisa::render {

struct alignas(16) TextureHandle {

    static constexpr auto texture_id_offset_shift = 6u;
    static constexpr auto tag_mask = (1 << texture_id_offset_shift) - 1u;
    static constexpr auto fixed_point_alpha_max = 4096.0f - 1.0f;
    static constexpr auto fixed_point_alpha_scale = 16384.0f;
    static constexpr auto tag_max_count = 1u << texture_id_offset_shift;

    uint id_and_tag;
    float compressed_v[3];

    [[nodiscard]] static TextureHandle encode_constant(uint tag, float3 v, float alpha = 1.0f) noexcept;
    [[nodiscard]] static TextureHandle encode_texture(uint tag, uint tex_id, float3 v = make_float3(1.0f)) noexcept;
};

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::TextureHandle, id_and_tag, compressed_v) {
    [[nodiscard]] auto tag() const noexcept {
        return id_and_tag & luisa::render::TextureHandle::tag_mask;
    }
    [[nodiscard]] auto v() const noexcept {
        return luisa::compute::def<luisa::float3>(compressed_v);
    }
    [[nodiscard]] auto alpha() const noexcept {
        using luisa::compute::cast;
        using luisa::render::TextureHandle;
        return cast<float>(id_and_tag >> TextureHandle::texture_id_offset_shift) *
            (1.0f / TextureHandle::fixed_point_alpha_scale);
    }
    [[nodiscard]] auto texture_id() const noexcept {
        return id_and_tag >> luisa::render::TextureHandle::texture_id_offset_shift;
    }
};
// clang-format on

namespace luisa::render {

class Pipeline;
class Interaction;
class SampledWavelengths;
using compute::Float4;

class Texture : public SceneNode {

private:
    friend class Pipeline;
    [[nodiscard]] virtual TextureHandle encode(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Texture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] uint handle_tag() const noexcept;
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual bool is_color() const noexcept = 0;     // automatically converts to the albedo spectrum
    [[nodiscard]] virtual bool is_general() const noexcept = 0;   // returns the value as-is (no conversion spectrum)
    [[nodiscard]] virtual bool is_illuminant() const noexcept = 0;// automatically converts to the illuminant spectrum
    [[nodiscard]] virtual Float4 evaluate(
        const Pipeline &pipeline, const Interaction &it,
        const Var<TextureHandle> &handle,
        const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
};

using compute::PixelStorage;
using TextureSampler = compute::Sampler;

class ImageTexture : public Texture {

private:
    TextureSampler _sampler;

public:
    ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
};

}// namespace luisa::render
