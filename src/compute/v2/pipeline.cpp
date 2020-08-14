//
// Created by Mike Smith on 2020/8/14.
//

#include "pipeline.h"

namespace luisa::compute {

Pipeline &Pipeline::append(PipelineStage *stage, uint3 threadgroups, uint3 threadgroup_size) noexcept {
    _stages.emplace_back(stage, threadgroups, threadgroup_size);
    return *this;
}

Pipeline &Pipeline::append(PipelineStage *stage, uint threadgroups, uint threadgroup_size) noexcept {
    return append(stage, make_uint3(threadgroups, 1u, 1u), make_int3(threadgroup_size, 1u, 1u));
}

Pipeline &Pipeline::append(PipelineStage *stage, uint2 threadgroups, uint2 threadgroup_size) noexcept {
    return append(stage, make_uint3(threadgroups, 1u), make_int3(threadgroup_size, 1u));
}

void Pipeline::run(Dispatcher &d) {
    for (auto[stage, tg, tg_size] : _stages) {
        stage->run(d, tg, tg_size);
    }
}

}
