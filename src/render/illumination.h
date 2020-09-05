//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <vector>

#include <compute/dsl.h>
#include <compute/dispatcher.h>
#include <compute/buffer.h>

#include "spectrum.h"
#include "interaction.h"
#include "data_block.h"

namespace luisa::render {

using compute::dsl::Var;
using compute::dsl::Expr;

using compute::Dispatcher;
using compute::BufferView;

class Shape;

struct LightSampleBuffers {
    BufferView<SampledSpectrum> L;
    BufferView<float> pdf;
};

struct LightSampleExpr {
    Var<SampledSpectrum> L;
    Var<float3> p;
    Var<packed_float3> w;
    Var<float> pdf;
};

struct Illumination {
    
    struct Description {
        std::vector<float> cdf;
        std::vector<float> pdf;
        float power;
    };
    
    [[nodiscard]] virtual Description description(const Shape *shape) = 0;
    
    [[nodiscard]] virtual LightSampleExpr sample(
        Expr<const DataBlock *> data,
        Expr<Interaction> p,
        Expr<SampledWavelength> lambda,
        Expr<float2> u) = 0;
};

}
