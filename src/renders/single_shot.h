//
// Created by Mike Smith on 2020/2/11.
//

#pragma once

#include <core/render.h>

namespace luisa {

class SingleShot : public Render {

protected:
    float _shutter_open;
    float _shutter_close;
    std::shared_ptr<Camera> _camera;
    std::filesystem::path _output_path_prefix;
    Viewport _viewport{};
    uint _command_queue_size;
    uint _working_command_count{0u};
    std::condition_variable _cv;
    std::mutex _mutex;

public:
    SingleShot(Device *device, const ParameterSet &parameter_set);
    void execute() override;
};

LUISA_REGISTER_NODE_CREATOR("SingleShot", SingleShot);

}
