//
// Created by Mike on 2021/12/15.
//

#include <scene/transform.h>

namespace luisa::render {

Transform::Transform(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TRANSFORM} {}

void TransformTree::Builder::push(const Transform *t) noexcept {
    auto current = _node_stack.back();
    auto node = current->add_child(t, ~0u, false, current->is_static());
    if (t != nullptr && !t->is_identity()) {
        if (!t->is_static()) {
            for (auto iter = _node_stack.rbegin(); iter != _node_stack.rend() && (*iter)->is_static(); iter++) {
                (*iter)->mark_dynamic();
            }
        }
        _transform_stack.emplace_back(
            _transform_stack.back() * t->matrix(_initial_time));
    }
    _node_stack.emplace_back(node);
}

inline TransformTree::Builder::Builder(float initial_time) noexcept
    : _tree{luisa::make_unique<TransformTree>(TransformTree{})},
      _initial_time{initial_time} {
    _node_stack.emplace_back(_tree->_root.get());
    _transform_stack.emplace_back(make_float4x4(1.0f));
}

void TransformTree::Builder::pop() noexcept {
    if (auto t = _node_stack.back()->transform();
        t != nullptr && !t->is_identity()) {
        _transform_stack.pop_back();
    }
    _node_stack.pop_back();
}

luisa::unique_ptr<TransformTree> TransformTree::Builder::build() noexcept {
    return std::move(_tree);
}

float4x4 TransformTree::Builder::leaf(const Transform *t, uint index) noexcept {
    auto current = _node_stack.back();
    current->add_child(t, index, true, current->is_static());
    auto matrix = _transform_stack.back();
    if (t != nullptr && !t->is_identity()) {
        if (!t->is_static()) {
            for (auto iter = _node_stack.rbegin(); iter != _node_stack.rend() && (*iter)->is_static(); iter++) {
                (*iter)->mark_dynamic();
            }
        }
        matrix = matrix * t->matrix(_initial_time);
    }
    return matrix;
}

void TransformTree::update(Accel &accel, float time) const noexcept {
    _root->update(accel, make_float4x4(1.0f), time);
}

TransformTree::Builder TransformTree::builder(float init_time) noexcept {
    return Builder{init_time};
}

inline TransformTree::TransformTree() noexcept
    : _root{luisa::make_unique<Node>(nullptr, ~0u, false, true)} {}

void TransformTree::Node::update(Accel &accel, float4x4 matrix, float time) const noexcept {
    if (is_static()) { return; }// static path in tree, prune
    if (_transform != nullptr) { matrix = matrix * _transform->matrix(time); }
    if (is_leaf()) {
        accel.set_transform(_transform_id, matrix);
    } else {
        for (auto &&child : _children) {
            child.update(accel, matrix, time);
        }
    }
}

inline TransformTree::Node *TransformTree::Node::add_child(
    const Transform *transform, uint32_t transform_id, bool is_leaf, bool ancestors_static) noexcept {
    _children.emplace_front(transform, transform_id, is_leaf, ancestors_static);
    return &_children.front();
}

inline TransformTree::Node::Node(
    const Transform *transform, uint32_t transform_id,
    bool is_leaf, bool ancestors_static) noexcept
    : _transform{transform}, _transform_id{transform_id}, _is_leaf{is_leaf},
      _is_static{ancestors_static && (transform == nullptr || transform->is_static())} {}

}// namespace luisa::render
