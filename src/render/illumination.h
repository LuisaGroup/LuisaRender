//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <vector>

#include <compute/dsl.h>
#include <compute/dispatcher.h>
#include <compute/buffer.h>

#include "interaction.h"
#include "data_block.h"

namespace luisa::render {

using compute::dsl::Var;
using compute::dsl::Expr;

using compute::Dispatcher;
using compute::BufferView;

class Shape;

struct LightSampleBuffers {
    BufferView<float3> L;
    BufferView<float> pdf;
};

struct LightSampleExpr {
    Expr<float3> L;
    Expr<float3> p;
    Expr<float3> w;
    Expr<float> pdf;
};

struct Illumination {
    
    struct Description {
        std::vector<float> cdf;
        std::vector<float> pdf;
        float power;
    };
    
    [[nodiscard]] virtual Description description(const Shape *shape) = 0;
};

}
