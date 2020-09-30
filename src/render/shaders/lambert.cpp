//
// Created by Mike Smith on 2020/9/18.
//

#include <render/sampling.h>
#include "lambert.h"

using namespace luisa::compute;
using namespace luisa::compute::dsl;

namespace luisa::render::shader {

struct alignas(16) LambertData {
    float r;
    float g;
    float b;
    bool double_sided;
};

}

LUISA_STRUCT(luisa::render::shader::LambertData, r, g, b, double_sided)

namespace luisa::render::shader {

class LambertReflection : public Surface<LambertReflection> {

public:
    using Data = LambertData;
    static constexpr auto is_emissive = false;

private:
    Data _data;

public:
    LambertReflection(float3 albedo, bool double_sided) noexcept: _data{albedo.x, albedo.y, albedo.z, double_sided} {}
    
    [[nodiscard]] static Expr<Scattering> evaluate(Expr<float2> uv [[maybe_unused]], Expr<float3> n, Expr<float3> wo, Expr<float3> wi,
                                                   Expr<float2> u2, Expr<Data> data, uint comp) {
        
        Var cos_i = dot(n, wi);
        Var cos_o = dot(n, wo);
        Var is_refl = cos_i * cos_o > 0.0f;
        Var is_front = cos_o > 0.0f;
        Var albedo = make_float3(data.r, data.g, data.b);
        Var double_sided = data.double_sided;
        
        Var<Scattering> scattering;
        if (comp & EVAL_BSDF) {
            Var valid = is_refl && (double_sided || is_front);
            Var f = select(valid, albedo * inv_pi, make_float3(0.0f));
            Var pdf = select(valid, abs(cos_i) * inv_pi, 0.0f);
            scattering.evaluation.f = f;
            scattering.evaluation.pdf = pdf;
        }
        if (comp & EVAL_BSDF_SAMPLING) {
            Var valid = double_sided || is_front;
            Var onb = make_onb(n);
            Var local_wi = cosine_sample_hemisphere(u2);
            Var sampled_wi = normalize(transform_to_world(onb, local_wi));
            sampled_wi.z *= sign(cos_o);
            Var sampled_f = select(valid, albedo * inv_pi, make_float3(0.0f));
            Var sampled_pdf = select(valid, abs(local_wi.z) * inv_pi, 0.0f);
            scattering.sample.wi = sampled_wi;
            scattering.sample.f = sampled_f;
            scattering.sample.pdf = sampled_pdf;
        }
        return scattering;
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
    LambertEmission(float3 e, bool double_sided) noexcept: _data{e.x, e.y, e.z, double_sided} {}
    
    [[nodiscard]] static Expr<Scattering> evaluate(
        Expr<float2> uv [[maybe_unused]], Expr<float3> n, Expr<float3> wo, Expr<float3> wi [[maybe_unused]],
        Expr<float2> u2 [[maybe_unused]], Expr<Data> data, uint comp) {
        
        Var<Scattering> scattering;
        if (comp & EVAL_EMISSION) {
            Var is_front = dot(wo, n);
            Var double_sided = data.double_sided;
            Var valid = double_sided || is_front;
            Var emission = select(valid, make_float3(data.r, data.g, data.b), make_float3(0.0f));
            Var pdf = select(valid, 1.0f, 0.0f);
            scattering.emission.L = emission;
        }
        return scattering;
    }
    
    [[nodiscard]] static Expr<Emission> emission(Expr<float2> uv [[maybe_unused]], Expr<float3> n, Expr<float3> w, Expr<Data> data) {
        Var is_front = dot(w, n) > 0.0f;
        Var double_sided = data.double_sided;
        Var valid = double_sided || is_front;
        Var<Emission> emission;
        emission.L = select(valid, make_float3(data.r, data.g, data.b), make_float3(0.0f));
        return emission;
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
