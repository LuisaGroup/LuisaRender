//
// Created by Mike on 2021/12/20.
//

#include <cstdint>
#include <string_view>

namespace luisa::render {

enum struct SceneNodeTag : uint32_t {
    ROOT,
    INTERNAL,
    DECLARATION,
    CAMERA,
    SHAPE,
    MATERIAL,
    LIGHT,
    TRANSFORM,
    FILM,
    FILTER,
    SAMPLER,
    INTEGRATOR,
    ENVIRONMENT
    // TODO: MEDIUM?
};

constexpr std::string_view scene_node_tag_description(SceneNodeTag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case SceneNodeTag::ROOT: return "__root__"sv;
        case SceneNodeTag::INTERNAL: return "__internal__"sv;
        case SceneNodeTag::DECLARATION: return "__declaration__"sv;
        case SceneNodeTag::CAMERA: return "Camera"sv;
        case SceneNodeTag::SHAPE: return "Shape"sv;
        case SceneNodeTag::MATERIAL: return "Material"sv;
        case SceneNodeTag::LIGHT: return "Light"sv;
        case SceneNodeTag::TRANSFORM: return "Transform"sv;
        case SceneNodeTag::FILM: return "Film"sv;
        case SceneNodeTag::FILTER: return "Filter"sv;
        case SceneNodeTag::SAMPLER: return "Sampler"sv;
        case SceneNodeTag::INTEGRATOR: return "Integrator"sv;
        case SceneNodeTag::ENVIRONMENT: return "Environment"sv;
        default: break;
    }
    return "__invalid__"sv;
}

}
