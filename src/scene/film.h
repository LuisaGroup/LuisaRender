//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <core/basic_types.h>
#include <scene/scene_node.h>

namespace luisa::render {

class Film : public SceneNode {

public:
    class Instance : public SceneNode::Instance {

    private:
        const Film *_film;

    public:
        explicit Instance(const Film *film) noexcept : _film{film} {}
        [[nodiscard]] auto node() const noexcept { return _film; }
        virtual void accumulate(Expr<uint2> pixel, Expr<float3> color) const noexcept = 0;
        virtual void clear(CommandBuffer &command_buffer) noexcept = 0;
        virtual void save(Stream &stream, const std::filesystem::path &path) const noexcept = 0;
    };

private:
    uint2 _resolution;

public:
    Film(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render
