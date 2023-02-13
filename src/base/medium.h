//
// Created by ChenXin on 2023/2/13.
//

#pragma once

#include <util/spec.h>
#include <util/scattering.h>
#include <base/scene_node.h>
#include <base/spectrum.h>
#include <base/interaction.h>

#include <utility>

namespace luisa::render {

using compute::Expr;
using compute::Var;

class Medium : public SceneNode {

public:
    class Instance;

    class Instance {

    protected:
        const Pipeline &_pipeline;
        const Medium *_medium;
        friend class Medium;

    public:
        Instance(const Pipeline &pipeline, const Medium *medium) noexcept
            : _pipeline{pipeline}, _medium{medium} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Medium>
            requires std::is_base_of_v<Medium, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_medium); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
    };

protected:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Medium(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;

};

}// namespace luisa::render