//
// Created by Mike Smith on 2022/10/13.
//

#include <array>

#include <util/colorspace.h>
#include <textures/sky_precompute.h>

namespace luisa::render {

// from Cycles: https://developer.blender.org/diffusion/B/browse/master/intern/sky/source/sky_nishita.cpp

/* SPDX-License-Identifier: Apache-2.0
* Copyright 2011-2020 Blender Foundation. */

/* Constants */
static constexpr auto rayleigh_scale = 8e3f;      // Rayleigh scale height (m)
static constexpr auto mie_scale = 1.2e3f;         // Mie scale height (m)
static constexpr auto mie_coeff = 2e-5f;          // Mie scattering coefficient (m^-1)
static constexpr auto mie_G = 0.76f;              // aerosols anisotropy
static constexpr auto sqr_G = mie_G * mie_G;      // squared aerosols anisotropy
static constexpr auto earth_radius = 6360e3f;     // radius of Earth (m)
static constexpr auto atmosphere_radius = 6420e3f;// radius of atmosphere (m)
static constexpr auto steps = 32u;                // segments of primary ray
static constexpr auto num_wavelengths = 21u;      // number of wavelengths
static constexpr auto min_wavelength = 380u;      // lowest sampled wavelength (nm)
static constexpr auto max_wavelength = 780u;      // highest sampled wavelength (nm)

// step between each sampled wavelength (nm)
static constexpr auto step_lambda = (max_wavelength - min_wavelength) / (num_wavelengths - 1);

/* Sun irradiance on top of the atmosphere (W*m^-2*nm^-1) */
static constexpr std::array irradiance = {
    1.45756829855592995315f, 1.56596305559738380175f, 1.65148449067670455293f,
    1.71496242737209314555f, 1.75797983805020541226f, 1.78256407885924539336f,
    1.79095108475838560302f, 1.78541550133410664714f, 1.76815554864306845317f,
    1.74122069647250410362f, 1.70647127164943679389f, 1.66556087452739887134f,
    1.61993437242451854274f, 1.57083597368892080581f, 1.51932335059305478886f,
    1.46628494965214395407f, 1.41245852740172450623f, 1.35844961970384092709f,
    1.30474913844739281998f, 1.25174963272610817455f, 1.19975998755420620867f};

/* Rayleigh scattering coefficient (m^-1) */
static constexpr std::array rayleigh_coeff = {
    0.00005424820087636473f, 0.00004418549866505454f, 0.00003635151910165377f,
    0.00003017929012024763f, 0.00002526320226989157f, 0.00002130859310621843f,
    0.00001809838025320633f, 0.00001547057129129042f, 0.00001330284977336850f,
    0.00001150184784075764f, 0.00000999557429990163f, 0.00000872799973630707f,
    0.00000765513700977967f, 0.00000674217203751443f, 0.00000596134125832052f,
    0.00000529034598065810f, 0.00000471115687557433f, 0.00000420910481110487f,
    0.00000377218381260133f, 0.00000339051255477280f, 0.00000305591531679811f};

/* Ozone absorption coefficient (m^-1) */
static constexpr std::array ozone_coeff = {
    0.00000000325126849861f, 0.00000000585395365047f, 0.00000001977191155085f,
    0.00000007309568762914f, 0.00000020084561514287f, 0.00000040383958096161f,
    0.00000063551335912363f, 0.00000096707041180970f, 0.00000154797400424410f,
    0.00000209038647223331f, 0.00000246128056164565f, 0.00000273551299461512f,
    0.00000215125863128643f, 0.00000159051840791988f, 0.00000112356197979857f,
    0.00000073527551487574f, 0.00000046450130357806f, 0.00000033096079921048f,
    0.00000022512612292678f, 0.00000014879129266490f, 0.00000016828623364192f};

/* CIE XYZ color matching functions */
static constexpr std::array cmf_xyz = {
    make_float3(0.00136800000f, 0.00003900000f, 0.00645000100f),
    make_float3(0.01431000000f, 0.00039600000f, 0.06785001000f),
    make_float3(0.13438000000f, 0.00400000000f, 0.64560000000f),
    make_float3(0.34828000000f, 0.02300000000f, 1.74706000000f),
    make_float3(0.29080000000f, 0.06000000000f, 1.66920000000f),
    make_float3(0.09564000000f, 0.13902000000f, 0.81295010000f),
    make_float3(0.00490000000f, 0.32300000000f, 0.27200000000f),
    make_float3(0.06327000000f, 0.71000000000f, 0.07824999000f),
    make_float3(0.29040000000f, 0.95400000000f, 0.02030000000f),
    make_float3(0.59450000000f, 0.99500000000f, 0.00390000000f),
    make_float3(0.91630000000f, 0.87000000000f, 0.00165000100f),
    make_float3(1.06220000000f, 0.63100000000f, 0.00080000000f),
    make_float3(0.85444990000f, 0.38100000000f, 0.00019000000f),
    make_float3(0.44790000000f, 0.17500000000f, 0.00002000000f),
    make_float3(0.16490000000f, 0.06100000000f, 0.00000000000f),
    make_float3(0.04677000000f, 0.01700000000f, 0.00000000000f),
    make_float3(0.01135916000f, 0.00410200000f, 0.00000000000f),
    make_float3(0.00289932700f, 0.00104700000f, 0.00000000000f),
    make_float3(0.00069007860f, 0.00024920000f, 0.00000000000f),
    make_float3(0.00016615050f, 0.00006000000f, 0.00000000000f),
    make_float3(0.00004150994f, 0.00001499000f, 0.00000000000f)};

/* Parameters for optical depth quadrature.
* See the comment in ray_optical_depth for more detail.
* Computed using sympy and following Python code:
* # from sympy.integrals.quadrature import gauss_laguerre
* # from sympy import exp
* # x, w = gauss_laguerre(8, 50)
* # xend = 25
* # print([(xi / xend).evalf(10) for xi in x])
* # print([(wi * exp(xi) / xend).evalf(10) for xi, wi in zip(x, w)])
*/
static constexpr auto quadrature_steps = 8u;
static constexpr std::array quadrature_nodes = {
    0.006811185292f,
    0.03614807107f,
    0.09004346519f,
    0.1706680068f,
    0.2818362161f,
    0.4303406404f,
    0.6296271457f,
    0.9145252695f};
static constexpr std::array quadrature_weights = {
    0.01750893642f,
    0.04135477391f,
    0.06678839063f,
    0.09507698807f,
    0.1283416365f,
    0.1707430204f,
    0.2327233347f,
    0.3562490486f};

static inline auto geographical_to_direction(float lat, float lon) noexcept {
    return make_float3(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
}

static inline auto spec_to_xyz(const float *spectrum) noexcept {
    auto xyz = make_float3(0.f);
    for (auto i = 0u; i < num_wavelengths; i++) {
        xyz.x += cmf_xyz[i][0] * spectrum[i];
        xyz.y += cmf_xyz[i][1] * spectrum[i];
        xyz.z += cmf_xyz[i][2] * spectrum[i];
    }
    return xyz * static_cast<float>(step_lambda);
}

/* Atmosphere volume models */
static inline auto density_rayleigh(float height) noexcept {
    return exp(-height / rayleigh_scale);
}

static inline auto density_mie(float height) noexcept {
    return exp(-height / mie_scale);
}

static inline auto density_ozone(float height) noexcept {
    auto den = 0.f;
    if (height >= 10000.f && height < 25000.f) {
        den = 1.f / 15000.f * height - 2.f / 3.f;
    } else if (height >= 25000.f && height < 40000.f) {
        den = -(1.f / 15000.f * height - 8.f / 3.f);
    }
    return den;
}

static inline auto phase_rayleigh(float mu) noexcept {
    return 3.f / (16.f * pi) * (1.f + mu * mu);
}

static inline auto phase_mie(float mu) noexcept {
    return (3.f * (1.f - sqr_G) * (1.f + mu * mu)) /
           (8.f * pi * (2.f + sqr_G) *
            pow((1.f + sqr_G - 2.f * mie_G * mu), 1.5f));
}

/* Intersection helpers */
static inline auto surface_intersection(float3 pos, float3 dir) noexcept {
    if (dir.z >= 0.f) { return false; }
    auto b = -2.f * dot(dir, -pos);
    auto c = dot(pos, pos) - earth_radius * earth_radius;
    return b * b - 4.f * c >= 0.f;
}

static inline auto atmosphere_intersection(float3 pos, float3 dir) noexcept {
    auto b = -2.f * dot(dir, -pos);
    auto c = dot(pos, pos) - atmosphere_radius * atmosphere_radius;
    auto t = (-b + sqrt(b * b - 4.f * c)) * .5f;
    return make_float3(pos.x + dir.x * t, pos.y + dir.y * t, pos.z + dir.z * t);
}

static inline auto ray_optical_depth(float3 ray_origin, float3 ray_dir) noexcept {
    /* This function computes the optical depth along a ray.
     * Instead of using classic ray marching, the code is based on Gauss-Laguerre quadrature,
     * which is designed to compute the integral of f(x)*exp(-x) from 0 to infinity.
     * This works well here, since the optical depth along the ray tends to decrease exponentially.
     * By setting f(x) = g(x) exp(x), the exponentials cancel out and we get the integral of g(x).
     * The nodes and weights used here are the standard n=6 Gauss-Laguerre values, except that
     * the exp(x) scaling factor is already included in the weights.
     * The parametrization along the ray is scaled so that the last quadrature node is still within
     * the atmosphere. */
    auto ray_end = atmosphere_intersection(ray_origin, ray_dir);
    auto ray_length = distance(ray_origin, ray_end);
    auto segment = ray_length * ray_dir;

    /* instead of tracking the transmission spectrum across all wavelengths directly,
     * we use the fact that the density always has the same spectrum for each type of
     * scattering, so we split the density into a constant spectrum and a factor and
     * only track the factors */
    auto optical_depth = make_float3(0.f);
    for (auto i = 0u; i < quadrature_steps; i++) {
        auto P = ray_origin + quadrature_nodes[i] * segment;
        /* height above sea level */
        auto height = length(P) - earth_radius;
        auto density = make_float3(
            density_rayleigh(height),
            density_mie(height),
            density_ozone(height));
        optical_depth += density * quadrature_weights[i];
    }
    return optical_depth * ray_length;
}

static inline void single_scattering(float3 ray_dir, float3 sun_dir, float3 ray_origin, float air_density,
                                     float dust_density, float ozone_density, float *r_spectrum) noexcept {

    /* this code computes single-inscattering along a ray through the atmosphere */
    auto ray_end = atmosphere_intersection(ray_origin, ray_dir);
    auto ray_length = distance(ray_origin, ray_end);

    /* to compute the inscattering, we step along the ray in segments and accumulate
     * the inscattering as well as the optical depth along each segment */
    auto segment_length = ray_length / steps;
    auto segment = segment_length * ray_dir;

    /* instead of tracking the transmission spectrum across all wavelengths directly,
     * we use the fact that the density always has the same spectrum for each type of
     * scattering, so we split the density into a constant spectrum and a factor and
     * only track the factors */
    auto optical_depth = make_float3(0.f);

    /* zero out light accumulation */
    for (auto wl = 0u; wl < num_wavelengths; wl++) {
        r_spectrum[wl] = 0.f;
    }

    /* phase function for scattering and the density scale factor */
    auto mu = dot(ray_dir, sun_dir);
    auto phase_function = make_float3(phase_rayleigh(mu), phase_mie(mu), 0.f);
    auto density_scale = make_float3(air_density, dust_density, ozone_density);

    /* the density and in-scattering of each segment is evaluated at its middle */
    auto P = ray_origin + .5f * segment;

    for (auto i = 0u; i < steps; i++) {
        /* height above sea level */
        auto height = length(P) - earth_radius;

        /* evaluate and accumulate optical depth along the ray */
        auto density = density_scale * make_float3(density_rayleigh(height),
                                                   density_mie(height),
                                                   density_ozone(height));
        optical_depth += segment_length * density;

        /* if the Earth isn't in the way, evaluate inscattering from the sun */
        if (!surface_intersection(P, sun_dir)) {
            auto light_optical_depth = density_scale * ray_optical_depth(P, sun_dir);
            auto total_optical_depth = optical_depth + light_optical_depth;

            /* attenuation of light */
            for (int wl = 0; wl < num_wavelengths; wl++) {
                auto extinction_density = total_optical_depth *
                                          make_float3(rayleigh_coeff[wl], 1.11f * mie_coeff, ozone_coeff[wl]);
                auto reduce_add = [](auto v) noexcept { return v.x + v.y + v.z; };
                auto attenuation = exp(-reduce_add(extinction_density));
                auto scattering_density = density * make_float3(rayleigh_coeff[wl], mie_coeff, 0.f);

                /* the total inscattered radiance from one segment is:
                 * Tr(A<->B) * Tr(B<->C) * sigma_s * phase * L * segment_length
                 *
                 * These terms are:
                 * Tr(A<->B): Transmission from start to scattering position (tracked in optical_depth)
                 * Tr(B<->C): Transmission from scattering position to light (computed in
                 * ray_optical_depth) sigma_s: Scattering density phase: Phase function of the scattering
                 * type (Rayleigh or Mie) L: Radiance coming from the light source segment_length: The
                 * length of the segment
                 *
                 * The code here is just that, with a bit of additional optimization to not store full
                 * spectra for the optical depth
                 */
                r_spectrum[wl] += attenuation * reduce_add(phase_function * scattering_density) *
                                  irradiance[wl] * segment_length;
            }
        }

        /* advance along ray */
        P += segment;
    }
}

/*********** Sun ***********/
static inline void sun_radiation(float3 cam_dir,
                                 float altitude,
                                 float air_density,
                                 float dust_density,
                                 float solid_angle,
                                 float *r_spectrum) noexcept {
    auto cam_pos = make_float3(0.f, 0.f, earth_radius + altitude);
    auto optical_depth = ray_optical_depth(cam_pos, cam_dir);

    /* compute final spectrum */
    for (auto i = 0u; i < num_wavelengths; i++) {
        /* combine spectra and the optical depth into transmittance */
        auto transmittance = rayleigh_coeff[i] * optical_depth.x * air_density +
                             1.11f * mie_coeff * optical_depth.y * dust_density;
        r_spectrum[i] = irradiance[i] * exp(-transmittance) / solid_angle;
    }
}

void SKY_nishita_skymodel_precompute_texture(NishitaSkyData data, float4 *pixels,
                                             uint2 resolution, uint2 y_range) noexcept {
    /* calculate texture pixels */
    float spectrum[num_wavelengths];
    auto cam_pos = make_float3(0.f, 0.f, earth_radius + data.altitude);
    auto sun_dir = geographical_to_direction(data.sun_elevation, 0.f);

    auto latitude_step = pi_over_two / static_cast<float>(resolution.y);
    auto longitude_step = 2.f * pi / static_cast<float>(resolution.x);
    auto half_lat_step = latitude_step * .5f;
    auto half_lon_step = longitude_step * .5f;
    for (auto y = y_range.x; y < y_range.y; y++) {
        /* sample more pixels toward the horizon */
        auto sqr = [](auto x) noexcept { return x * x; };
        // difference from Cycles: add .5f to y to sample the center of the pixel
        auto latitude = (pi_over_two + half_lat_step) *
                        sqr((static_cast<float>(y) + .5f) / static_cast<float>(resolution.y));
        auto pixel_row = pixels + y * resolution.x;
        // difference from Cycles: we do not store the other half of the
        // texture, so we use the image width rather than a half of it and
        // multiply x by half the longitude step
        for (auto x = 0u; x < resolution.x; x++) {
            // difference from Cycles: add .5f to x to sample the center of the pixel
            auto longitude = half_lon_step * (static_cast<float>(x) + .5f);
            auto dir = geographical_to_direction(latitude, longitude);
            single_scattering(dir, sun_dir, cam_pos, data.air_density,
                              data.dust_density, data.ozone_density, spectrum);
            auto rgb = cie_xyz_to_linear_srgb(spec_to_xyz(spectrum));
            pixel_row[x] = make_float4(rgb, 1.f);
        }
    }
}

NishitaSkyPrecomputedSun
SKY_nishita_skymodel_precompute_sun(NishitaSkyData data) noexcept {
    /* definitions */
    auto half_angular = data.sun_angle / 2.f;
    auto solid_angle = 2.f * pi * (1.f - cos(half_angular));
    float spectrum[num_wavelengths];
    auto bottom = data.sun_elevation - half_angular;
    auto top = data.sun_elevation + half_angular;

    /* compute 2 pixels for sun disc */
    auto elevation_bottom = (bottom > 0.f) ? bottom : 0.f;
    auto elevation_top = (top > 0.f) ? top : 0.f;
    auto sun_dir = geographical_to_direction(elevation_bottom, 0.f);
    sun_radiation(sun_dir, data.altitude, data.air_density,
                  data.dust_density, solid_angle, spectrum);
    auto pix_bottom = cie_xyz_to_linear_srgb(spec_to_xyz(spectrum));
    sun_dir = geographical_to_direction(elevation_top, 0.f);
    sun_radiation(sun_dir, data.altitude, data.air_density,
                  data.dust_density, solid_angle, spectrum);
    auto pix_top = cie_xyz_to_linear_srgb(spec_to_xyz(spectrum));
    return {.bottom = pix_bottom, .top = pix_top};
}

}// namespace luisa::render
