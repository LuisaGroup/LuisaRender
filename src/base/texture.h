//
// Created by Mike Smith on 2022/1/25.
//

#pragma once

#include <dsl/syntax.h>
#include <util/spectrum.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/scene_node.h>

namespace luisa::render {

struct alignas(16) TextureHandle {

    static constexpr auto texture_id_offset_shift = 8u;
    static constexpr auto tag_mask = (1 << texture_id_offset_shift) - 1u;
    static constexpr auto tag_max_count = 1u << texture_id_offset_shift;

    uint id_and_tag;
    float compressed_v[3];

    [[nodiscard]] static TextureHandle encode_constant(
        uint tag, float3 v, float alpha = 1.0f) noexcept;
    [[nodiscard]] static TextureHandle encode_texture(
        uint tag, uint tex_id, float3 v = make_float3(1.0f)) noexcept;
};

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::TextureHandle, id_and_tag, compressed_v) {
    [[nodiscard]] auto tag() const noexcept {
        return id_and_tag & luisa::render::TextureHandle::tag_mask;
    }
    [[nodiscard]] auto v() const noexcept {
        return luisa::compute::make_float3(
            compressed_v[0],
            compressed_v[1],
            compressed_v[2]);
    }
    [[nodiscard]] auto alpha() const noexcept {
        using luisa::compute::cast;
        using luisa::render::TextureHandle;
        return luisa::render::half_to_float(
            id_and_tag >> TextureHandle::texture_id_offset_shift);
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

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Texture *_texture;

    public:
        Instance(const Pipeline &pipeline, const Texture *texture) noexcept
            : _pipeline{pipeline}, _texture{texture} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Texture>
            requires std::is_base_of_v<Texture, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_texture); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Float4 evaluate(const Interaction &it, Expr<float> time) const noexcept = 0;
    };

private:
    friend class Pipeline;
    [[nodiscard]] virtual TextureHandle _encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint handle_tag) const noexcept = 0;

public:
    Texture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual uint channels() const noexcept { return 4u; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

using compute::PixelStorage;
using TextureSampler = compute::Sampler;

class ImageTexture : public Texture {

private:
    TextureSampler _sampler;
    float2 _uv_scale;
    float2 _uv_offset;

private:
    [[nodiscard]] virtual const LoadedImage &_image() const noexcept = 0;
    [[nodiscard]] TextureHandle _encode(
        Pipeline &pipeline, CommandBuffer &command_buffer, uint handle_tag) const noexcept override;

public:
    ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }
    [[nodiscard]] Float4 evaluate(
        const Pipeline &pipeline, const Interaction &it,
        const Var<TextureHandle> &handle, Expr<float> time) const noexcept override;
};

}// namespace luisa::render
