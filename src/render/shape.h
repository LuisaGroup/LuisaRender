//
// Created by Mike Smith on 2020/9/4.
//

#pragma once

#include <vector>

#include <core/data_types.h>
#include <render/parser.h>
#include <render/material.h>
#include <render/transform.h>

namespace luisa::render {

using compute::Vertex;
using compute::TriangleHandle;
using compute::EntityHandle;

class Shape : public Plugin {

protected:
    std::vector<Vertex> _vertices;
    std::vector<TriangleHandle> _triangles;
    std::shared_ptr<Material> _material;
    std::shared_ptr<Transform> _transform;
    std::vector<std::shared_ptr<Shape>> _children;

private:
    bool _cleared{false};
    void _exception_if_cleared() const { LUISA_EXCEPTION_IF(_cleared, "Invalid operation on cleared shape."); }

public:
    Shape(Device *device, const ParameterSet &params) noexcept
        : Plugin{device, params},
          _material{params["material"].parse_or_null<Material>()},
          _transform{params["transform"].parse_or_null<Transform>()} {}
    
    [[nodiscard]] const std::vector<Vertex> &vertices() const {
        _exception_if_cleared();
        return _vertices;
    }
    
    [[nodiscard]] const std::vector<TriangleHandle> &triangles() const {
        _exception_if_cleared();
        return _triangles;
    }
    
    [[nodiscard]] const std::vector<std::shared_ptr<Shape>> &children() const noexcept { return _children; }
    
    void clear() noexcept {  // to save some memory...
        _vertices.clear();
        _triangles.clear();
        _vertices.shrink_to_fit();
        _triangles.shrink_to_fit();
        for (auto &&child : _children) { child->clear(); }
        _cleared = true;
    }
    
    [[nodiscard]] bool is_entity() const noexcept { return _children.empty(); }
    [[nodiscard]] Transform *transform() const noexcept { return _transform.get(); }
    [[nodiscard]] Material *material() const noexcept { return _material.get(); }
};

}
