//
// Created by Mike Smith on 2022/9/27.
//

#include <core/stl.h>
#include <sdl/scene_node_tag.h>

namespace luisa::render {

SceneNodeTag parse_scene_node_tag(luisa::string_view tag_desc) noexcept {
    luisa::string tag{tag_desc};
    for (auto &c : tag) { c = static_cast<char>(std::tolower(c)); }
    using namespace std::string_view_literals;
    static constexpr auto desc_to_tag_count = 28u;
    static const luisa::fixed_map<std::string_view, SceneNodeTag, desc_to_tag_count> desc_to_tag{
        {"camera"sv, SceneNodeTag::CAMERA},
        {"cam"sv, SceneNodeTag::CAMERA},
        {"shape"sv, SceneNodeTag::SHAPE},
        {"object"sv, SceneNodeTag::SHAPE},
        {"obj"sv, SceneNodeTag::SHAPE},
        {"surface"sv, SceneNodeTag::SURFACE},
        {"surf"sv, SceneNodeTag::SURFACE},
        {"lightsource"sv, SceneNodeTag::LIGHT},
        {"light"sv, SceneNodeTag::LIGHT},
        {"illuminant"sv, SceneNodeTag::LIGHT},
        {"illum"sv, SceneNodeTag::LIGHT},
        {"transform"sv, SceneNodeTag::TRANSFORM},
        {"xform"sv, SceneNodeTag::TRANSFORM},
        {"film"sv, SceneNodeTag::FILM},
        {"filter"sv, SceneNodeTag::FILTER},
        {"sampler"sv, SceneNodeTag::SAMPLER},
        {"integrator"sv, SceneNodeTag::INTEGRATOR},
        {"lightsampler"sv, SceneNodeTag::LIGHT_SAMPLER},
        {"environment"sv, SceneNodeTag::ENVIRONMENT},
        {"env"sv, SceneNodeTag::ENVIRONMENT},
        {"texture"sv, SceneNodeTag::TEXTURE},
        {"tex"sv, SceneNodeTag::TEXTURE},
        {"texturemapping"sv, SceneNodeTag::TEXTURE_MAPPING},
        {"texmapping"sv, SceneNodeTag::TEXTURE_MAPPING},
        {"spectrum"sv, SceneNodeTag::SPECTRUM},
        {"spec"sv, SceneNodeTag::SPECTRUM},
        {"generic"sv, SceneNodeTag::DECLARATION},
        {"template"sv, SceneNodeTag::DECLARATION}};
    if (auto iter = desc_to_tag.find(tag); iter != desc_to_tag.end()) {
        return iter->second;
    }
    return SceneNodeTag::ROOT;
}

}
