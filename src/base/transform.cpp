//
// Created by Mike on 2021/12/15.
//

#include <base/transform.h>

namespace luisa::render {

Transform::Transform(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TRANSFORM} {}

TransformTree::Node::Node(
    const TransformTree::Node *parent,
    const Transform *t) noexcept
    : _parent{parent}, _transform{t} {}

float4x4 TransformTree::Node::matrix(float time) const noexcept {
    auto m = _transform->matrix(time);
    for (auto node = _parent; node != nullptr; node = node->_parent) {
        m = node->_transform->matrix(time) * m;
    }
    return m;
}

TransformTree::TransformTree() noexcept {
    _node_stack.emplace_back(nullptr);
    _static_stack.emplace_back(true);
}

void TransformTree::push(const Transform *t) noexcept {
    if (t != nullptr && !t->is_identity()) {
        auto node = luisa::make_unique<Node>(_node_stack.back(), t);
        _node_stack.emplace_back(_nodes.emplace_back(std::move(node)).get());
        _static_stack.emplace_back(_static_stack.back() && t->is_static());
    }
}

void TransformTree::pop(const Transform *t) noexcept {
    if (t != nullptr && !t->is_identity()) {
        assert(
            !_node_stack.empty() &&
            _node_stack.back()->transform() == t);
        _node_stack.pop_back();
        _static_stack.pop_back();
    }
}

std::pair<const TransformTree::Node *, bool> TransformTree::leaf(
    const Transform *t) noexcept {
    if (t == nullptr || t->is_identity()) {
        return std::make_pair(
            _node_stack.back(),
            _static_stack.back());
    }
    auto node = luisa::make_unique<Node>(_node_stack.back(), t);
    auto p_node = _nodes.emplace_back(std::move(node)).get();
    auto is_static = _static_stack.back() && t->is_static();
    return std::make_pair(p_node, is_static);
}

}// namespace luisa::render
