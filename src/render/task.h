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
        LUISA_INFO("Compiling & running pipeline...");
        auto t0 = std::chrono::high_resolution_clock::now();
        _compile(_pipeline);
        _pipeline << compute::synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        LUISA_INFO("Rendering time: ", (t1 - t0) / 1ns * 1e-9, "s");
    }
};

}
