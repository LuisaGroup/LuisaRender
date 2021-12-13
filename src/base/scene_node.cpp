//
// Created by Mike on 2021/12/13.
//

#include <base/scene_node.h>

namespace luisa::render {

const SceneNode *SceneNode::child(size_t index) const noexcept {
    return const_cast<SceneNode *>(this)->child(index);
}

}
