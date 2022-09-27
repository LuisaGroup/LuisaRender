//
// Created by Mike Smith on 2022/9/27.
//

#include <core/stl.h>
#include <sdl/scene_node_tag.h>

namespace luisa::render {

SceneNodeTag parse_scene_node_tag(std::string_view tag) noexcept {
    using namespace std::string_view_literals;
    static constexpr auto desc_to_tag_count = 28u;
    static const luisa::fixed_map<std::string_view, SceneNodeTag, desc_to_tag_count> desc_to_tag{
        {"Camera"sv, SceneNodeTag::CAMERA},
        {"Cam"sv, SceneNodeTag::CAMERA},
        {"Shape"sv, SceneNodeTag::SHAPE},
        {"Object"sv, SceneNodeTag::SHAPE},
        {"Obj"sv, SceneNodeTag::SHAPE},
        {"Surface"sv, SceneNodeTag::SURFACE},
        {"Surf"sv, SceneNodeTag::SURFACE},
        {"LightSource"sv, SceneNodeTag::LIGHT},
        {"Light"sv, SceneNodeTag::LIGHT},
        {"Illuminant"sv, SceneNodeTag::LIGHT},
        {"Illum"sv, SceneNodeTag::LIGHT},
        {"Transform"sv, SceneNodeTag::TRANSFORM},
        {"Xform"sv, SceneNodeTag::TRANSFORM},
        {"Film"sv, SceneNodeTag::FILM},
        {"Filter"sv, SceneNodeTag::FILTER},
        {"Sampler"sv, SceneNodeTag::SAMPLER},
        {"Integrator"sv, SceneNodeTag::INTEGRATOR},
        {"LightSampler"sv, SceneNodeTag::LIGHT_SAMPLER},
        {"Environment"sv, SceneNodeTag::ENVIRONMENT},
        {"Env"sv, SceneNodeTag::ENVIRONMENT},
        {"Texture"sv, SceneNodeTag::TEXTURE},
        {"Tex"sv, SceneNodeTag::TEXTURE},
        {"TextureMapping"sv, SceneNodeTag::TEXTURE_MAPPING},
        {"TexMapping"sv, SceneNodeTag::TEXTURE_MAPPING},
        {"Spectrum"sv, SceneNodeTag::SPECTRUM},
        {"Spec"sv, SceneNodeTag::SPECTRUM},
        {"Generic"sv, SceneNodeTag::DECLARATION},
        {"Template"sv, SceneNodeTag::DECLARATION}};
    if (auto iter = desc_to_tag.find(tag); iter != desc_to_tag.end()) {
        return iter->second;
    }
    return SceneNodeTag::ROOT;
}

}