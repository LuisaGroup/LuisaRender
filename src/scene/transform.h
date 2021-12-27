//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl.h>
#include <rtx/accel.h>
#include <runtime/buffer.h>
#include <scene/scene_node.h>

namespace luisa::render {

class Transform : public SceneNode {
public:
    Transform(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual bool is_identity() const noexcept = 0;
    [[nodiscard]] virtual float4x4 matrix(float time) const noexcept = 0;
};

using compute::Accel;

class TransformTree {

private:
    class Node {

    private:
        const Transform *_transform;
        luisa::slist<Node> _children;
        uint32_t _transform_id;
        bool _is_leaf;
        bool _is_static;

    public:
        Node(const Transform *transform, uint32_t transform_id, bool is_leaf, bool ancestors_static) noexcept;
        Node(Node &&) noexcept = delete;
        Node(const Node &) noexcept = delete;
        Node &operator=(Node &&) noexcept = delete;
        Node &operator=(const Node &) noexcept = delete;
        void mark_dynamic() noexcept { _is_static = false; }
        [[nodiscard]] auto transform() const noexcept { return _transform; }
        [[nodiscard]] auto is_leaf() const noexcept { return _is_leaf; }
        [[nodiscard]] auto is_static() const noexcept { return _is_static; }
        Node *add_child(const Transform *transform, uint32_t transform_id, bool is_leaf, bool ancestors_static) noexcept;
        void update(Accel &accel, float4x4 matrix, float time) const noexcept;
    };

public:
    class Builder {

    private:
        luisa::unique_ptr<TransformTree> _tree;
        luisa::vector<Node *> _node_stack;
        luisa::vector<float4x4> _transform_stack;
        float _initial_time;

    public:
        explicit Builder(float initial_time = 0.0f) noexcept;
        void push(const Transform *t) noexcept;
        void pop() noexcept;
        [[nodiscard]] float4x4 leaf(const Transform *t, uint index) noexcept;
        [[nodiscard]] luisa::unique_ptr<TransformTree> build() noexcept;
    };

private:
    luisa::unique_ptr<Node> _root;

private:
    TransformTree() noexcept;

public:
    TransformTree(TransformTree &&) noexcept = default;
    [[nodiscard]] static Builder builder(float init_time = 0.0f) noexcept;
    void update(Accel &accel, float time) const noexcept;
    [[nodiscard]] auto is_static() const noexcept { return _root->is_static(); }
};

}// namespace luisa::render
