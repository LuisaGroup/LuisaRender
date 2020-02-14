//
// Created by Mike Smith on 2020/2/11.
//

#pragma once

#include <core/render.h>
#include <core/integrator.h>
#include <core/shape.h>
#include <core/light.h>
#include <core/geometry.h>
#include <core/camera.h>
#include <core/illumination.h>

namespace luisa {

class SingleShot : public Render {

public:
    SingleShot(Device *device, const ParameterSet &parameter_set);
    void execute() override;
};

LUISA_REGISTER_NODE_CREATOR("SingleShot", SingleShot);

}
