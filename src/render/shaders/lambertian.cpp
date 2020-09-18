//
// Created by Mike Smith on 2020/9/18.
//

#include <render/sampling.h>
#include "lambertian.h"

using namespace luisa::compute;
using namespace luisa::compute::dsl;

namespace luisa::render {

struct alignas(16) LambertianReflectionData {
    float r;
    float g;
    float b;
    bool double_sided;
};

}

LUISA_STRUCT(luisa::render::LambertianReflectionData, r, g, b, double_sided)

namespace luisa::render {

class LambertianReflection : public Surface<LambertianReflection> {

public:
    using Data = LambertianReflectionData;
    static constexpr auto is_emissive = false;

private:
    Data _data;

public:
    LambertianReflection(float3 albedo, bool double_sided) noexcept : _data{albedo.x, albedo.y, albedo.z, double_sided} {}
    
    [[nodiscard]] static SurfaceEvaluation evaluate(Expr<float2> uv, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<Data> data) {
        Var is_refl = wi.z * wo.z > 0.0f;
        Var is_front = wo.z > 0.0f;
        Var albedo = make_float3(data.r, data.g, data.b);
        Var double_sided = data.double_sided;
        Var valid = double_sided || is_front;
        Var f = select(is_refl && valid, albedo * inv_pi, make_float3(0.0f));
        Var pdf = select(is_refl && valid, abs(wi.z) * inv_pi, 0.0f);
        Var sampled_wi = sign(wi.z) * cosine_sample_hemisphere(u2);
        Var sampled_f = select(valid, albedo * inv_pi, make_float3(0.0f));
        Var sampled_pdf = select(valid, abs(sampled_wi.z) * inv_pi, 0.0f);
        return SurfaceEvaluation{make_float3(0.0f), f, pdf, sampled_wi, sampled_f, sampled_pdf};
    }
    
    [[nodiscard]] const Data &data() const noexcept { return _data; }
};

std::unique_ptr<SurfaceShader> create_lambertian_reflection(float3 albedo, bool double_sided) {
    return std::make_unique<LambertianReflection>(albedo, double_sided);
}

}
