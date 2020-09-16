//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <algorithm>
#include <vector>
#include <memory>

#include <compute/dsl.h>
#include <render/plugin.h>

namespace luisa::render {

struct Transform : public Plugin {
    
    Transform(Device *device, const ParameterSet &params) noexcept: Plugin{device, params} {}
    
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual float4x4 matrix(float time) const noexcept = 0;
};

class TransformTree {

private:
    const Transform *_transform{nullptr};
    std::vector<std::unique_ptr<TransformTree>> _children;
    uint _instance_id{};
    bool _is_leaf{false};

public:
    void add_leaf(const Transform *transform, uint instance_id) noexcept {
        auto child = std::make_unique<TransformTree>();
        child->_transform = transform;
        child->_instance_id = instance_id;
        child->_is_leaf = true;
        _children.emplace_back(std::move(child));
    }
    
    [[nodiscard]] TransformTree *add_inner_node(const Transform *transform) noexcept {
        auto child = std::make_unique<TransformTree>();
        child->_transform = transform;
        auto child_ptr = child.get();
        _children.emplace_back(std::move(child));
        return child_ptr;
    }
    
    void update(float4x4 *buffer, float time, float4x4 parent_matrix = make_float4x4(1.0f)) {
        auto m = _transform == nullptr ? parent_matrix : parent_matrix * _transform->matrix(time);
        if (_is_leaf) {
            buffer[_instance_id] = m;
        } else {
            for (auto &&child : _children) {
                child->update(buffer, time, m);
            }
        }
    }
    
    [[nodiscard]] bool is_static() const noexcept {
        return (_transform == nullptr || _transform->is_static()) &&
               std::all_of(_children.cbegin(), _children.cend(), [](auto &&child) { return child->is_static(); });
    }
};

}
