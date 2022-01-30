#pragma once

#include "cuda_utils.h"
#include "Float3.h"
#include "lcg_rng.h"

/* Disney BSDF functions, for additional details and examples see:
 * - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
 * - https://www.shadertoy.com/view/XdyyDd
 * - https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
 * - https://schuttejoe.github.io/post/disneybsdf/
 *
 * Variable naming conventions with the Burley course notes:
 * V -> w_o
 * L -> w_i
 * H -> w_h
 */

struct DisneySurfaceClosure {
    Float3 base_color;
    Float metallic;

    Float specular;
    Float roughness;
    Float specular_tint;
    Float anisotropy;

    Float sheen;
    Float sheen_tint;
    Float clearcoat;
    Float clearcoat_gloss;

    Float ior;
    Float specular_transmission;
};

__device__ bool same_hemisphere(const Float3 &w_o, const Float3 &w_i, const Float3 &n) {
    return dot(w_o, n) * dot(w_i, n) > 0.f;
}

// Sample the hemisphere using a cosine weighted distribution,
// returns a vector in a hemisphere oriented about (0, 0, 1)
__device__ Float3 cos_sample_hemisphere(Float2 u) {
    Float2 s = 2.f * u - make_float2(1.f);
    Float2 d;
    Float radius = 0.f;
    Float theta = 0.f;
    if (s.x == 0.f && s.y == 0.f) {
        d = s;
    } else {
        if (abs(s.x) > abs(s.y)) {
            radius = s.x;
            theta  = M_PI / 4.f * (s.y / s.x);
        } else {
            radius = s.y;
            theta  = M_PI / 2.f - M_PI / 4.f * (s.x / s.y);
        }
    }
    d = radius * make_float2(cos(theta), sin(theta));
    return make_float3(d.x, d.y, sqrt(max(0.f, 1.f - d.x * d.x - d.y * d.y)));
}

__device__ Float3 spherical_dir(Float sin_theta, Float cos_theta, Float phi) {
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

__device__ Float power_heuristic(Float n_f, Float pdf_f, Float n_g, Float pdf_g) {
    Float f = n_f * pdf_f;
    Float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

__device__ Float schlick_weight(Float cos_theta) {
    return pow(saturate(1.f - cos_theta), 5.f);
}

// Complete Fresnel Dielectric computation, for transmission at ior near 1
// they mention having issues with the Schlick approximation.
// eta_i: material on incident side's ior
// eta_t: material on transmitted side's ior
__device__ Float fresnel_dielectric(Float cos_theta_i, Float eta_i, Float eta_t) {
    Float g = pow2(eta_t) / pow2(eta_i) - 1.f + pow2(cos_theta_i);
    if (g < 0.f) {
        return 1.f;
    }
    return 0.5f * pow2(g - cos_theta_i) / pow2(g + cos_theta_i)
           * (1.f + pow2(cos_theta_i * (g + cos_theta_i) - 1.f) /
                        pow2(cos_theta_i * (g - cos_theta_i) + 1.f));
}

// D_GTR1: Generalized Trowbridge-Reitz with gamma=1
// Burley notes eq. 4
__device__ Float gtr_1(Float cos_theta_h, Float alpha) {
    if (alpha >= 1.f) {
        return inv_pi;
    }
    Float alpha_sqr = alpha * alpha;
    return inv_pi * (alpha_sqr - 1.f) / (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
}

// D_GTR2: Generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 8
__device__ Float gtr_2(Float cos_theta_h, Float alpha) {
    Float alpha_sqr = alpha * alpha;
    return inv_pi * alpha_sqr /
           pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 13
__device__ Float gtr_2_aniso(Float h_dot_n, Float h_dot_x, Float h_dot_y, Float2 alpha) {
    return inv_pi / (alpha.x * alpha.y
                     * pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n));
}

__device__ Float smith_shadowing_ggx(Float n_dot_o, Float alpha_g) {
    Float a = alpha_g * alpha_g;
    Float b = n_dot_o * n_dot_o;
    return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

__device__ Float smith_shadowing_ggx_aniso(Float n_dot_o, Float o_dot_x, Float o_dot_y, Float2 alpha) {
    return 1.f / (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
}

// Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
__device__ Float3 sample_lambertian_dir(const Float3 &n, const Float3 &v_x, const Float3 &v_y, const Float2 &s) {
    const Float3 hemi_dir = normalize(cos_sample_hemisphere(s));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

// Sample the microfacet normal vectors for the various microfacet distributions
__device__ Float3 sample_gtr_1_h(const Float3 &n, const Float3 &v_x, const Float3 &v_y, Float alpha, const Float2 &s) {
    Float phi_h = 2.f * M_PI * s.x;
    Float alpha_sqr = alpha * alpha;
    Float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
    Float cos_theta_h = sqrt(cos_theta_h_sqr);
    Float sin_theta_h = 1.f - cos_theta_h_sqr;
    Float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ Float3 sample_gtr_2_h(const Float3 &n, const Float3 &v_x, const Float3 &v_y, Float alpha, const Float2 &s) {
    Float phi_h = 2.f * M_PI * s.x;
    Float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
    Float cos_theta_h = sqrt(cos_theta_h_sqr);
    Float sin_theta_h = 1.f - cos_theta_h_sqr;
    Float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ Float3 sample_gtr_2_aniso_h(const Float3 &n, const Float3 &v_x, const Float3 &v_y, const Float2 &alpha, const Float2 &s) {
    Float x = 2.f * M_PI * s.x;
    Float3 w_h = sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
    return normalize(w_h);
}

__device__ Float lambertian_pdf(const Float3 &w_i, const Float3 &n) {
    Float d = dot(w_i, n);
    if (d > 0.f) {
        return d * inv_pi;
    }
    return 0.f;
}

__device__ Float gtr_1_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n, Float alpha) {
    if (!same_hemisphere(w_o, w_i, n)) {
        return 0.f;
    }
    Float3 w_h = normalize(w_i + w_o);
    Float cos_theta_h = dot(n, w_h);
    Float d = gtr_1(cos_theta_h, alpha);
    return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ Float gtr_2_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n, Float alpha) {
    if (!same_hemisphere(w_o, w_i, n)) {
        return 0.f;
    }
    Float3 w_h = normalize(w_i + w_o);
    Float cos_theta_h = dot(n, w_h);
    Float d = gtr_2(cos_theta_h, alpha);
    return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ Float gtr_2_transmission_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n,
                                        Float alpha, Float ior)
{
    if (same_hemisphere(w_o, w_i, n)) {
        return 0.f;
    }
    bool entering = dot(w_o, n) > 0.f;
    Float eta_o = entering ? 1.f : ior;
    Float eta_i = entering ? ior : 1.f;
    Float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
    Float cos_theta_h = abs(dot(n, w_h));
    Float i_dot_h = dot(w_i, w_h);
    Float o_dot_h = dot(w_o, w_h);
    Float d = gtr_2(cos_theta_h, alpha);
    Float dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
    return d * cos_theta_h * abs(dwh_dwi);
}

__device__ Float gtr_2_aniso_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n,
                                 const Float3 &v_x, const Float3 &v_y, const Float2 alpha)
{
    if (!same_hemisphere(w_o, w_i, n)) {
        return 0.f;
    }
    Float3 w_h = normalize(w_i + w_o);
    Float cos_theta_h = dot(n, w_h);
    Float d = gtr_2_aniso(cos_theta_h, abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
    return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ Float3 disney_diffuse(const DisneySurfaceClosure &mat, const Float3 &n,
                                 const Float3 &w_o, const Float3 &w_i)
{
    Float3 w_h = normalize(w_i + w_o);
    Float n_dot_o = abs(dot(w_o, n));
    Float n_dot_i = abs(dot(w_i, n));
    Float i_dot_h = dot(w_i, w_h);
    Float fd90 = 0.5f + 2.f * mat.roughness * i_dot_h * i_dot_h;
    Float fi = schlick_weight(n_dot_i);
    Float fo = schlick_weight(n_dot_o);
    return mat.base_color * inv_pi * lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo);
}

__device__ Float3 disney_microfacet_isotropic(const DisneySurfaceClosure &mat, const Float3 &n,
                                              const Float3 &w_o, const Float3 &w_i)
{
    Float3 w_h = normalize(w_i + w_o);
    Float lum = luminance(mat.base_color);
    Float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
    Float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

    Float alpha = max(0.001f, mat.roughness * mat.roughness);
    Float d = gtr_2(dot(n, w_h), alpha);
    Float3 f = lerp(spec, make_float3(1.f), schlick_weight(dot(w_i, w_h)));
    Float g = smith_shadowing_ggx(dot(n, w_i), alpha) * smith_shadowing_ggx(dot(n, w_o), alpha);
    return d * f * g;
}

__device__ Float3 disney_microfacet_transmission_isotropic(const DisneySurfaceClosure &mat, const Float3 &n,
                                                           const Float3 &w_o, const Float3 &w_i)
{
    Float o_dot_n = dot(w_o, n);
    Float i_dot_n = dot(w_i, n);
    if (o_dot_n == 0.f || i_dot_n == 0.f) {
        return make_float3(0.f);
    }
    bool entering = o_dot_n > 0.f;
    Float eta_o = entering ? 1.f : mat.ior;
    Float eta_i = entering ? mat.ior : 1.f;
    Float3 w_h = normalize(w_o + w_i * eta_i / eta_o);

    Float alpha = max(0.001f, mat.roughness * mat.roughness);
    Float d = gtr_2(abs(dot(n, w_h)), alpha);

    Float f = fresnel_dielectric(abs(dot(w_i, n)), eta_o, eta_i);
    Float g = smith_shadowing_ggx(abs(dot(n, w_i)), alpha) * smith_shadowing_ggx(abs(dot(n, w_o)), alpha);

    Float i_dot_h = dot(w_i, w_h);
    Float o_dot_h = dot(w_o, w_h);

    Float c = abs(o_dot_h) / abs(dot(w_o, n)) * abs(i_dot_h) / abs(dot(w_i, n))
              * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);

    return mat.base_color * c * (1.f - f) * g * d;
}

__device__ Float3 disney_microfacet_anisotropic(const DisneySurfaceClosure &mat, const Float3 &n,
                                                const Float3 &w_o, const Float3 &w_i, const Float3 &v_x, const Float3 &v_y)
{
    Float3 w_h = normalize(w_i + w_o);
    Float lum = luminance(mat.base_color);
    Float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
    Float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

    Float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
    Float a = mat.roughness * mat.roughness;
    Float2 alpha = make_float2(max(0.001f, a / aspect), max(0.001f, a * aspect));
    Float d = gtr_2_aniso(dot(n, w_h), abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
    Float3 f = lerp(spec, make_float3(1.f), schlick_weight(dot(w_i, w_h)));
    Float g = smith_shadowing_ggx_aniso(dot(n, w_i), abs(dot(w_i, v_x)), abs(dot(w_i, v_y)), alpha)
              * smith_shadowing_ggx_aniso(dot(n, w_o), abs(dot(w_o, v_x)), abs(dot(w_o, v_y)), alpha);
    return d * f * g;
}

__device__ Float disney_clear_coat(const DisneySurfaceClosure &mat, const Float3 &n,
                                   const Float3 &w_o, const Float3 &w_i)
{
    Float3 w_h = normalize(w_i + w_o);
    Float alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
    Float d = gtr_1(dot(n, w_h), alpha);
    Float f = lerp(0.04f, 1.f, schlick_weight(dot(w_i, n)));
    Float g = smith_shadowing_ggx(dot(n, w_i), 0.25f) * smith_shadowing_ggx(dot(n, w_o), 0.25f);
    return 0.25f * mat.clearcoat * d * f * g;
}

__device__ Float3 disney_sheen(const DisneySurfaceClosure &mat, const Float3 &n,
                               const Float3 &w_o, const Float3 &w_i)
{
    Float3 w_h = normalize(w_i + w_o);
    Float lum = luminance(mat.base_color);
    Float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
    Float3 sheen_color = lerp(make_float3(1.f), tint, mat.sheen_tint);
    Float f = schlick_weight(dot(w_i, n));
    return f * mat.sheen * sheen_color;
}

__device__ Float3 disney_brdf(const DisneySurfaceClosure &mat, const Float3 &n,
                              const Float3 &w_o, const Float3 &w_i, const Float3 &v_x, const Float3 &v_y)
{
    if (!same_hemisphere(w_o, w_i, n)) {
        if (mat.specular_transmission > 0.f) {
            Float3 spec_trans = disney_microfacet_transmission_isotropic(mat, n, w_o, w_i);
            return spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
        }
        return make_float3(0.f);
    }

    Float coat = disney_clear_coat(mat, n, w_o, w_i);
    Float3 sheen = disney_sheen(mat, n, w_o, w_i);
    Float3 diffuse = disney_diffuse(mat, n, w_o, w_i);
    Float3 gloss;
    if (mat.anisotropy == 0.f) {
        gloss = disney_microfacet_isotropic(mat, n, w_o, w_i);
    } else {
        gloss = disney_microfacet_anisotropic(mat, n, w_o, w_i, v_x, v_y);
    }
    return (diffuse + sheen) * (1.f - mat.metallic) * (1.f - mat.specular_transmission) + gloss + coat;
}

__device__ Float disney_pdf(const DisneySurfaceClosure &mat, const Float3 &n,
                            const Float3 &w_o, const Float3 &w_i, const Float3 &v_x, const Float3 &v_y)
{
    Float alpha = max(0.001f, mat.roughness * mat.roughness);
    Float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
    Float2 alpha_aniso = make_float2(max(0.001f, alpha / aspect), max(0.001f, alpha * aspect));

    Float clearcoat_alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);

    Float diffuse = lambertian_pdf(w_i, n);
    Float clear_coat = gtr_1_pdf(w_o, w_i, n, clearcoat_alpha);

    Float n_comp = 3.f;
    Float microfacet = 0.f;
    Float microfacet_transmission = 0.f;
    if (mat.anisotropy == 0.f) {
        microfacet = gtr_2_pdf(w_o, w_i, n, alpha);
    } else {
        microfacet = gtr_2_aniso_pdf(w_o, w_i, n, v_x, v_y, alpha_aniso);
    }
    if (mat.specular_transmission > 0.f) {
        n_comp = 4.f;
        microfacet_transmission = gtr_2_transmission_pdf(w_o, w_i, n, alpha, mat.ior);
    }
    return (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
}

/* Sample a component of the Disney BRDF, returns the sampled BRDF color,
 * ray reflection direction (w_i) and sample PDF.
 */
__device__ Float3 sample_disney_brdf(const DisneySurfaceClosure &mat, const Float3 &n,
                                     const Float3 &w_o, const Float3 &v_x, const Float3 &v_y, LCGRand &rng,
                                     Float3 &w_i, Float &pdf)
{
    int component = 0;
    if (mat.specular_transmission == 0.f) {
        component = lcg_randomf(rng) * 3.f;
        component = clamp(component, 0, 2);
    } else {
        component = lcg_randomf(rng) * 4.f;
        component = clamp(component, 0, 3);
    }

    Float2 samples = make_float2(lcg_randomf(rng), lcg_randomf(rng));
    if (component == 0) {
        // Sample diffuse component
        w_i = sample_lambertian_dir(n, v_x, v_y, samples);
    } else if (component == 1) {
        Float3 w_h;
        Float alpha = max(0.001f, mat.roughness * mat.roughness);
        if (mat.anisotropy == 0.f) {
            w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples);
        } else {
            Float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
            Float2 alpha_aniso = make_float2(max(0.001f, alpha / aspect), max(0.001f, alpha * aspect));
            w_h = sample_gtr_2_aniso_h(n, v_x, v_y, alpha_aniso, samples);
        }
        w_i = reflect(-w_o, w_h);

        // Invalid reflection, terminate ray
        if (!same_hemisphere(w_o, w_i, n)) {
            pdf = 0.f;
            w_i = make_float3(0.f);
            return make_float3(0.f);
        }
    } else if (component == 2) {
        // Sample clear coat component
        Float alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
        Float3 w_h = sample_gtr_1_h(n, v_x, v_y, alpha, samples);
        w_i = reflect(-w_o, w_h);

        // Invalid reflection, terminate ray
        if (!same_hemisphere(w_o, w_i, n)) {
            pdf = 0.f;
            w_i = make_float3(0.f);
            return make_float3(0.f);
        }
    } else {
        // Sample microfacet transmission component
        Float alpha = max(0.001f, mat.roughness * mat.roughness);
        Float3 w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples);
        if (dot(w_o, w_h) < 0.f) {
            w_h = -w_h;
        }
        bool entering = dot(w_o, n) > 0.f;
        w_i = refract_ray(-w_o, w_h, entering ? 1.f / mat.ior : mat.ior);

        // Invalid refraction, terminate ray
        if (all_zero(w_i)) {
            pdf = 0.f;
            return make_float3(0.f);
        }
    }
    pdf = disney_pdf(mat, n, w_o, w_i, v_x, v_y);
    return disney_brdf(mat, n, w_o, w_i, v_x, v_y);
}
