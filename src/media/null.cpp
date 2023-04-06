//
// Created by ChenXin on 2023/2/13.
//

#include <base/medium.h>
#include <base/pipeline.h>

namespace luisa::render {

using compute::Ray;

class NullMedium : public Medium {

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return nullptr;
    }

public:
    NullMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc} {
        _priority = 0u;
    }
    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NullMedium)