//
// Created by Mike Smith on 2022/9/14.
//

#pragma once

#include <dsl/syntax.h>
#include <util/command_buffer.h>

namespace luisa::render {

class Pipeline;
using compute::Float;
using compute::Expr;

class SPD {

private:
    const Pipeline &_pipeline;
    uint _buffer_id;
    float _sample_interval;

public:
    SPD(Pipeline &pipeline, uint buffer_id, float sample_interval) noexcept;
    [[nodiscard]] static SPD create_cie_x(Pipeline &pipeline, CommandBuffer &cb) noexcept;
    [[nodiscard]] static SPD create_cie_y(Pipeline &pipeline, CommandBuffer &cb) noexcept;
    [[nodiscard]] static SPD create_cie_z(Pipeline &pipeline, CommandBuffer &cb) noexcept;
    [[nodiscard]] static SPD create_cie_d65(Pipeline &pipeline, CommandBuffer &cb) noexcept;
    [[nodiscard]] static float cie_y_integral() noexcept;
    [[nodiscard]] Float sample(Expr<float> lambda) const noexcept;
};

}// namespace luisa::render
