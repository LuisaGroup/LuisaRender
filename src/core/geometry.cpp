//
// Created by Mike Smith on 2020/2/10.
//

#include "geometry.h"
#include "shape.h"

namespace luisa {

void Geometry::add(std::shared_ptr<Shape> shape) {
    shape->load(GeometryEncoder{this});
    _shapes.emplace_back(std::move(shape));
}

}
