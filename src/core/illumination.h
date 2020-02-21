//
// Created by Mike Smith on 2020/2/21.
//

#pragma once

#include "light.h"
#include "geometry.h"

namespace luisa {

class Illumination {

private:
    Geometry *_geometry;
    std::vector<std::shared_ptr<Light>> _lights;

public:
    Illumination(Device *device, const std::vector<std::shared_ptr<Light>> &lights, Geometry *geometry /* in case there are mesh lights */);

};

}
