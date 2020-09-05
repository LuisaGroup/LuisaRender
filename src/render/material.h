//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/type_desc.h>
#include <compute/buffer.h>

#include "plugin.h"
#include "illumination.h"
#include "material_handle.h"

namespace luisa::render {

using compute::dsl::Var;
using compute::dsl::Expr;

struct MaterialSampleExpr {
    Var<SampledSpectrum> f;
    Var<float3> wi;
    Var<float> pdf;
    Var<bool> is_specular;
    Var<bool> is_trans;
};

struct MaterialEvaluationExpr {
    Var<SampledSpectrum> f;
    Var<SampledSpectrum> L;
    Var<float> pdf_bsdf;
    Var<float> pdf_light;
};

class Material : public Plugin, public Illumination {

public:


};

}
