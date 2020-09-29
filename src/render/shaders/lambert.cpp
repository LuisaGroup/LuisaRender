//
// Created by Mike Smith on 2020/9/18.
//

#include <render/sampling.h>
#include "lambert.h"

using namespace luisa::compute;
using namespace luisa::compute::dsl;

namespace luisa::render::shading {

struct alignas(16) LambertData {
    float r;
    float g;
    float b;
    bool double_sided;
};

}

LUISA_STRUCT(luisa::render::shading::LambertData, r, g, b, double_sided)

namespace luisa::render::shading {

class LambertReflection : public Surface<LambertReflection> {

public:
    using Data = LambertData;
    static constexpr auto is_emissive = false;

private:
    Data _data;

public:
    LambertReflection(float3 albedo, bool double_sided) noexcept : _data{albedo.x, albedo.y, albedo.z, double_sided} {}
    
    [[nodiscard]] static Scattering evaluate(Expr<float2> uv [[maybe_unused]], Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<Data> data) {
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
        return Scattering{make_float3(0.0f), 0.0f, f, pdf, sampled_wi, sampled_f, sampled_pdf};
    }
    
    [[nodiscard]] const Data &data() const noexcept { return _data; }
};

class LambertEmission : public Surface<LambertEmission> {

public:
    using Data = LambertData;
    static constexpr auto is_emissive = true;

private:
    Data _data;

public:
    LambertEmission(float3 e, bool double_sided) noexcept : _data{e.x, e.y, e.z, double_sided} {}
    
    [[nodiscard]] static Scattering evaluate(Expr<float2> uv [[maybe_unused]], Expr<float3> wo, Expr<float3> wi [[maybe_unused]], Expr<float2> u2 [[maybe_unused]], Expr<Data> data) {
        Var is_front = wo.z > 0.0f;
        Var double_sided = data.double_sided;
        Var valid = double_sided || is_front;
        Var emission = select(valid, make_float3(data.r, data.g, data.b), make_float3(0.0f));
        Var pdf = select(valid, 1.0f, 0.0f);
        return Scattering{emission, pdf, make_float3(0.0f), 0.0f, make_float3(0.0f), make_float3(0.0f), 0.0f};
    }
    
    [[nodiscard]] static Emission emission(Expr<float2> uv [[maybe_unused]], Expr<float3> wo, Expr<Data> data) {
        Var is_front = wo.z > 0.0f;
        Var double_sided = data.double_sided;
        Var valid = double_sided || is_front;
        Var emission = select(valid, make_float3(data.r, data.g, data.b), make_float3(0.0f));
        Var pdf = select(valid, 1.0f, 0.0f);
        return Emission{emission, pdf};
    }
    
    [[nodiscard]] const Data &data() const noexcept { return _data; }
};

std::unique_ptr<SurfaceShader> create_lambert_reflection(float3 albedo, bool double_sided) {
    return std::make_unique<LambertReflection>(albedo, double_sided);
}

std::unique_ptr<SurfaceShader> create_lambert_emission(float3 emission, bool double_sided) {
    return std::make_unique<LambertEmission>(emission, double_sided);
}

}
