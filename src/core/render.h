//
// Created by Mike Smith on 2020/2/4.
//

#pragma once

#include <chrono>

#include "mathematics.h"
#include "node.h"
#include "parser.h"
#include "scene.h"
#include "camera.h"
#include "sampler.h"
#include "integrator.h"

namespace luisa {

class Render : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Render);

protected:
    std::shared_ptr<Sampler> _sampler;
    std::shared_ptr<Integrator> _integrator;
    std::unique_ptr<Scene> _scene;
    
    virtual void _execute() = 0;

public:
    Render(Device *device, const ParameterSet &parameter_set[[maybe_unused]])
        : Node{device},
          _sampler{parameter_set["sampler"].parse<Sampler>()},
          _integrator{parameter_set["integrator"].parse<Integrator>()} {}
    
    void execute() noexcept {
        auto t0 = std::chrono::high_resolution_clock::now();
        try {
            _execute();
        } catch (const std::runtime_error &e) {
            LUISA_WARNING("error occurred, render terminated, reason:\n    ", e.what());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        std::cout << "render time: " << (t1 - t0) / 1ns * 1e-9 << "s" << std::endl;
    }
};

}
