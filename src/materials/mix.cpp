//
// Created by Mike Smith on 2020/10/4.
//

#include <render/parser.h>
#include <render/material.h>

namespace luisa::render::material {

using namespace luisa::compute;

struct MixMaterial : public Material {
    MixMaterial(Device *device, const ParameterSet &params) : Material{device, params} {
        auto components = params["components"].parse_reference_list<Material>();
        LUISA_EXCEPTION_IF(components.empty(), "No components in MixMaterial.");
        auto weights = params["weights"].parse_float_list();
        LUISA_EXCEPTION_IF_NOT(components.size() == weights.size(), "Numbers of components and weights mismatch.");
        LUISA_WARNING_IF(std::accumulate(weights.cbegin(), weights.cend(), 0.0f) > 1.0f, "MixMaterial weights sum up to more than 1.");
        for (auto c = 0u; c < components.size(); c++) {
            for (auto &&lobe : components[c]->lobes()) {
                _lobes.emplace_back(lobe.shader, weights[c] * lobe.weight);
            }
        }
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::material::MixMaterial)
