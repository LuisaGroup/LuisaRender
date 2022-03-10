//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spectrum.h>
#include <base/scene_node.h>

namespace luisa::render {

class Film : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Film *_film;

    public:
        explicit Instance(const Pipeline &pipeline, const Film *film) noexcept
            : _pipeline{pipeline}, _film{film} {}
        virtual ~Instance() noexcept = default;

        template<typename T = Film>
            requires std::is_base_of_v<Film, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_film); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        virtual void accumulate(Expr<uint2> pixel, Expr<float3> rgb) const noexcept = 0;// TODO: spectrum
        virtual void clear(CommandBuffer &command_buffer) noexcept = 0;
        virtual void save(Stream &stream, const std::filesystem::path &path) const noexcept = 0;
    };

private:
    uint2 _resolution;

public:
    Film(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render
