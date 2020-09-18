//
// Created by Mike Smith on 2020/9/18.
//

#include <render/sampling.h>
#include "lambertian.h"

using namespace luisa::compute;
using namespace luisa::compute::dsl;

namespace luisa::render {

struct alignas(16) LambertianData {
    float r;
    float g;
    float b;
    bool double_sided;
};

}

LUISA_STRUCT(luisa::render::LambertianData, r, g, b, double_sided)

namespace luisa::render {

class Lambertian : public Surface<Lambertian> {

public:
    using Data = LambertianData;
    static constexpr auto is_emissive = false;

private:
    Data _data;

public:
    Lambertian(float3 albedo, bool double_sided) noexcept : _data{albedo.x, albedo.y, albedo.z, double_sided} {}
    
    [[nodiscard]] static SurfaceEvaluation evaluate(Expr<float2> uv, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<Data> data) {
        Var is_refl = wi.z * wo.z > 0.0f;
        Var albedo = make_float3(data.r, data.g, data.b);
        Var pdf = select(is_refl, abs(wi.z), 0.0f);
        Var sampled_wi = sign(wi.z) * cosine_sample_hemisphere(u2);
        Var sampled_pdf = select(is_refl, abs(wi.z), 0.0f);
        return SurfaceEvaluation{make_float3(0.0f), albedo, pdf, sampled_wi, albedo, sampled_pdf};
    }
    
    [[nodiscard]] const Data &data() const noexcept { return _data; }
};

std::unique_ptr<SurfaceShader> create_lambertian(float3 albedo, bool double_sided) {
    return std::make_unique<Lambertian>(albedo, double_sided);
}

}
