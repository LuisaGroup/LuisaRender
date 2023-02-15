//
// Created by Mike on 2021/12/20.
//

#pragma once

#include <core/stl.h>

namespace luisa::render {

enum struct SceneNodeTag : uint32_t {
    ROOT,
    INTERNAL,
    DECLARATION,
    CAMERA,
    SHAPE,
    SURFACE,
    LIGHT,
    TRANSFORM,
    FILM,
    FILTER,
    SAMPLER,
    INTEGRATOR,
    LIGHT_SAMPLER,
    ENVIRONMENT,
    TEXTURE,
    TEXTURE_MAPPING,
    SPECTRUM,
    MEDIUM,
    PHASE_FUNCTION,
};

constexpr std::string_view scene_node_tag_description(SceneNodeTag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case SceneNodeTag::ROOT: return "__root__"sv;
        case SceneNodeTag::INTERNAL: return "__internal__"sv;
        case SceneNodeTag::DECLARATION: return "__declaration__"sv;
        case SceneNodeTag::CAMERA: return "Camera"sv;
        case SceneNodeTag::SHAPE: return "Shape"sv;
        case SceneNodeTag::SURFACE: return "Surface"sv;
        case SceneNodeTag::LIGHT: return "Light"sv;
        case SceneNodeTag::TRANSFORM: return "Transform"sv;
        case SceneNodeTag::FILM: return "Film"sv;
        case SceneNodeTag::FILTER: return "Filter"sv;
        case SceneNodeTag::SAMPLER: return "Sampler"sv;
        case SceneNodeTag::INTEGRATOR: return "Integrator"sv;
        case SceneNodeTag::LIGHT_SAMPLER: return "LightSampler"sv;
        case SceneNodeTag::ENVIRONMENT: return "Environment"sv;
        case SceneNodeTag::TEXTURE: return "Texture"sv;
        case SceneNodeTag::TEXTURE_MAPPING: return "TextureMapping"sv;
        case SceneNodeTag::SPECTRUM: return "Spectrum"sv;
        case SceneNodeTag::MEDIUM: return "Medium"sv;
        case SceneNodeTag::PHASE_FUNCTION: return "PhaseFunction"sv;
        default: break;
    }
    return "__invalid__"sv;
}

[[nodiscard]] SceneNodeTag parse_scene_node_tag(luisa::string_view tag_desc) noexcept;

}// namespace luisa::render
