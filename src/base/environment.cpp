//
// Created by Mike Smith on 2022/1/14.
//

#include <base/environment.h>
#include <base/pipeline.h>

namespace luisa::render {

[[nodiscard]] static auto default_transform_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__environment_default_transform__", SceneNodeTag::TRANSFORM};
        d.define(SceneNodeTag::TRANSFORM, "Identity", {});
        return &d;
    }();
    return desc;
}

Environment::Environment(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::ENVIRONMENT},
      _transform{scene->load_transform(desc->property_node_or_default(
          "transform", default_transform_node_desc()))},
      _importance{std::max(desc->property_float_or_default("importance", 1.0f), 0.1f)} {}

Environment::Instance::Instance(Pipeline &pipeline, const Environment *env) noexcept
    : _pipeline{pipeline}, _env{env},
      _select_prob{env->importance() /
                   (env->importance() +
                    static_cast<float>(pipeline.lights().size()))} {}

}// namespace luisa::render
