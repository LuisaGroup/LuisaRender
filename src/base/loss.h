//
// Created by ChenXin on 2022/4/11.
//

#pragma once

#include <luisa-compute.h>

#include <base/camera.h>
#include <base/interaction.h>
#include <base/scene_node.h>
#include <base/spectrum.h>

namespace luisa::render {

using namespace luisa::compute;

class Loss : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Loss *_loss;

    public:
        Instance(const Pipeline &pipeline, const Loss *loss) noexcept
            : _pipeline{pipeline}, _loss{loss} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Loss>
            requires std::is_base_of_v<Loss, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_loss); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Float3 loss(const Camera::Instance *camera) const noexcept = 0;
        [[nodiscard]] virtual Float3 d_loss(const Camera::Instance *camera, Expr<uint2> pixel_id) const noexcept = 0;
    };

public:
    Loss(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

[[nodiscard]] Interaction pixel_xy2uv(Expr<uint2> pixel_id, uint2 resolution) noexcept;

}// namespace luisa::render