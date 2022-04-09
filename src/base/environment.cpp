//
// Created by Mike Smith on 2022/1/14.
//

#include <base/environment.h>
#include <base/pipeline.h>

namespace luisa::render {

Environment::Environment(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::ENVIRONMENT},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))},
      _importance{std::max(desc->property_float_or_default("importance", 1.0f), 0.01f)} {}

Environment::Instance::Instance(Pipeline &pipeline, const Environment *env) noexcept
    : _pipeline{pipeline}, _env{env} { pipeline.register_transform(env->transform()); }

Float3x3 Environment::Instance::transform_to_world() const noexcept {
    return make_float3x3(pipeline().transform(node()->transform()));
}

}// namespace luisa::render
