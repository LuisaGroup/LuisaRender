//
// Created by Mike Smith on 2020/9/29.
//

#include <render/parser.h>
#include <render/material.h>
#include <render/shaders/lambert.h>

namespace luisa::render::material {

using namespace luisa::compute;

struct LambertMaterial : public Material {
    
    LambertMaterial(Device *device, const ParameterSet &params) : Material{device, params} {
        auto albedo = clamp(params["albedo"].parse_float3_or_default(make_float3(0.0f)), 0.0f, 1.0f);
        auto emission = max(params["emission"].parse_float3_or_default(make_float3(0.0f)), 0.0f);
        auto double_sided = params["double_sided"].parse_bool_or_default(false);
        LUISA_EXCEPTION_IF(all(albedo == 0.0f && emission == 0.0f), "No valid lobe in LambertMaterial.");
        if (any(albedo != 0.0f)) { _lobes.emplace_back(shader::create_lambert_reflection(albedo, double_sided), 1.0f); }
        if (any(emission != 0.0f)) { _lobes.emplace_back(shader::create_lambert_emission(emission, double_sided), 1.0f); }
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::material::LambertMaterial)
