//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <vector>
#include <tuple>

#include <core/data_types.h>
#include <compute/v2/buffer.h>
#include <compute/v2/dispatcher.h>

namespace luisa::compute {

struct PipelineStage : Noncopyable {
    virtual ~PipelineStage() noexcept = default;
    virtual void run(Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) = 0;
};

class Pipeline {

private:
    std::vector<std::tuple<PipelineStage *, uint3, uint3>> _stages;

public:
    Pipeline &append(PipelineStage *stage, uint threadgroups, uint threadgroup_size) noexcept;
    Pipeline &append(PipelineStage *stage, uint2 threadgroups, uint2 threadgroup_size) noexcept;
    Pipeline &append(PipelineStage *stage, uint3 threadgroups, uint3 threadgroup_size) noexcept;
    void run(Dispatcher &d);
};

}
