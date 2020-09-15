//
// Created by Mike Smith on 2020/9/14.
//

#pragma once

#include <compute/pipeline.h>
#include <render/plugin.h>
#include <render/parser.h>

namespace luisa::render {

using compute::Pipeline;

class Task : public Plugin {

private:
    Pipeline _pipeline;
    
private:
    virtual void _compile(Pipeline &pipeline) = 0;

public:
    Task(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _pipeline{device} {}
    
    void execute() {
        _compile(_pipeline);
        _pipeline.run();
    }
};

}
