//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <vector>

#include <compute/v2/buffer.h>
#include <compute/v2/dispatcher.h>

namespace luisa::compute {

struct PipelineStage : Noncopyable {
    virtual ~PipelineStage() noexcept = default;
    virtual void run(Dispatcher &dispatcher) = 0;
};

class Pipeline {

private:
    std::vector<std::unique_ptr<PipelineStage>> _stages;

public:
    void add(std::unique_ptr<PipelineStage> stage) noexcept { _stages.emplace_back(std::move(stage)); }
    void run(Dispatcher &d) { std::for_each(_stages.begin(), _stages.end(), [&d](auto &&stage) { stage->run(d); }); }
    
    Pipeline &operator<<(std::unique_ptr<PipelineStage> stage) noexcept {
        add(std::move(stage));
        return *this;
    }
    
    void operator()(Dispatcher &dispatcher) { run(dispatcher); }
};

}
