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

namespace luisa::render {

struct LightSample {
    SampledSpectrum  L;
    packed_float3 wi;
    float pdf;
};

struct LightEvaluation {
    SampledSpectrum L;
    float pdf;
};

}

LUISA_STRUCT(luisa::render::LightSample, L, wi, pdf)
LUISA_STRUCT(luisa::render::LightEvaluation, L, pdf)

namespace luisa::render {

using compute::dsl::Expr;

using compute::Dispatcher;
using compute::BufferView;

class Shape;

struct LightSampleBuffers {
    BufferView<SampledSpectrum> L;
    BufferView<float4> wi_and_pdf;
};

class Illumination {

public:
    struct Description {
        std::vector<float> cdf;
        float power;
    };

private:
    virtual Expr<LightSample> _sample(Expr<Interaction> pi, Expr<SampledWavelength> lambda, Expr<float2> u) = 0;
    virtual Expr<LightEvaluation> _evaluate(Expr<float3> p, Expr<float3> n, Expr<float2> uv, Expr<float3> w, Expr<SampledWavelength> lambda) = 0;

public:
    [[nodiscard]] virtual Description description(const Shape *shape) = 0;
    // TODO: sample and eval interfaces
};

}
