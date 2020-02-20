//
// Created by Mike Smith on 2020/2/11.
//

#pragma once

#include <vector>
#include <random>
#include <filesystem>

#include <core/render.h>

namespace luisa {

class SingleShot : public Render {

protected:
    float _shutter_open;
    float _shutter_close;
    std::shared_ptr<Camera> _camera;
    std::filesystem::path _output_path_prefix;
    Viewport _viewport{};
    
    void _execute() override;
    
public:
    SingleShot(Device *device, const ParameterSet &parameter_set);
};

LUISA_REGISTER_NODE_CREATOR("SingleShot", SingleShot);

}
