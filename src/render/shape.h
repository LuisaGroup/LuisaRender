//
// Created by Mike Smith on 2020/9/4.
//

#pragma once

#include <vector>
#include <span>

#include <core/data_types.h>

#include "material.h"
#include "transform.h"

namespace luisa::render {

class Shape {

public:
    struct Vertex {
        packed_float3 position;
        packed_float3 normal;
        float2 uv;
    };

protected:
    std::vector<Vertex> _vertices;
    std::vector<uint3> _indices;
    std::shared_ptr<Material> _material;
    std::shared_ptr<Transform> _transform;
    std::vector<std::shared_ptr<Shape>> _children;

private:
    bool _cleared{false};
    void _exception_if_cleared() const { LUISA_EXCEPTION_IF(_cleared, "Invalid operation on cleared shape."); }

public:
    [[nodiscard]] const std::vector<Vertex> &vertices() const {
        _exception_if_cleared();
        return _vertices;
    }
    
    [[nodiscard]] const std::vector<uint3> &indices() const {
        _exception_if_cleared();
        return _indices;
    }
    
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &children() const noexcept { return _children; }
    
    void clear() noexcept {  // to save some memory...
        _vertices.clear();
        _indices.clear();
        _vertices.shrink_to_fit();
        _indices.shrink_to_fit();
        for (auto &&child : _children) { child->clear(); }
        _cleared = true;
    }
    
    [[nodiscard]] Transform *transform() const noexcept { return _transform.get(); }
    [[nodiscard]] Material *material() const noexcept { return _material.get(); }
};

}
