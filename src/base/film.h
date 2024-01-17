//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spec.h>
#include <base/scene_node.h>

namespace luisa::render {

class Film : public SceneNode {

public:
    struct Accumulation {
        Float3 average;
        Float sample_count;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Film *_film;

    protected:
        virtual void _accumulate(Expr<uint2> pixel, Expr<float3> rgb,
                                 Expr<float> effective_spp) const noexcept = 0;

    public:
        explicit Instance(const Pipeline &pipeline, const Film *film) noexcept
            : _pipeline{pipeline}, _film{film} {}

        virtual ~Instance() noexcept = default;
        template<typename T = Film>
            requires std::is_base_of_v<Film, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_film); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Accumulation read(Expr<uint2> pixel) const noexcept = 0;
        void accumulate(Expr<uint2> pixel, Expr<float3> rgb, Expr<float> effective_spp = 1.f) const noexcept;
        virtual void prepare(CommandBuffer &command_buffer) noexcept = 0;
        virtual void clear(CommandBuffer &command_buffer) noexcept = 0;
        virtual void download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept = 0;
        virtual bool show(CommandBuffer &command_buffer) const noexcept { return false; }
        virtual void *export_image(CommandBuffer &command_buffer) { return nullptr; }
        virtual void release() noexcept = 0;
    };

public:
    Film(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual uint2 resolution() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
    [[nodiscard]] virtual float clamp() const noexcept { return 1024.0f; }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Film::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(::luisa::render::Film::Accumulation)
