//
// Created by Mike on 2021/12/14.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/interaction.h>

namespace luisa::render {

Surface::Surface(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SURFACE},
      _normal_map{scene->load_texture(desc->property_node_or_default(
          "normal", SceneNodeDesc::shared_default(
                        SceneNodeTag::TEXTURE, "ConstGeneric")))} {}

Frame Surface::apply_normal_mapping(const Frame &f, Expr<float3> n_map) noexcept {
    using compute::all;
    using compute::ite;
    auto normal = ite(
        all(n_map == 0.f),
        make_float3(0.f, 0.f, 1.f),
        normalize(n_map * 2.f - 1.f) * make_float3(1.f, -1.f, 1.f));
    return Frame::make(f.local_to_world(normal), f.u());
}

}// namespace luisa::render
