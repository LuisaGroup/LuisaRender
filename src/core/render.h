//
// Created by Mike Smith on 2020/2/4.
//

#pragma once

#include <vector>
#include <random>
#include <filesystem>
#include <condition_variable>

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

public:
    Render(Device *device, const ParameterSet &parameter_set[[maybe_unused]])
        : Node{device},
          _sampler{parameter_set["sampler"].parse<Sampler>()},
          _integrator{parameter_set["integrator"].parse<Integrator>()} {}
    
    virtual void execute() = 0;
    
};

}
