//
// Created by Mike Smith on 2022/3/11.
//

#include <base/texture_mapping.h>

namespace luisa::render {

TextureMapping::TextureMapping(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE_MAPPING} {}

}// namespace luisa::render
