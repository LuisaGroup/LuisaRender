//
// Created by Mike Smith on 2022/9/26.
//

#pragma once

#include <string_view>
#include <filesystem>
#include <future>

#include <core/stl.h>
#include <core/json_fwd.h>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

class SceneDesc;
class SceneNodeDesc;

class SceneParserJSON {

public:
    using MacroMap = luisa::map<luisa::string, luisa::string>;

private:
    SceneDesc &_desc;
    const MacroMap &_cli_macros;
    SceneNodeDesc::SourceLocation _location;

private:
    void _parse_root(const json &root) const noexcept;
    void _parse_import(const json &node) const noexcept;
    void _parse_node(SceneNodeDesc &desc, const json &node) const noexcept;
    const SceneNodeDesc *_reference(luisa::string_view name) const noexcept;

public:
    SceneParserJSON(SceneDesc &desc, const std::filesystem::path &path,
                    const MacroMap &cli_macros) noexcept;
    void parse() const noexcept;
};

}// namespace luisa::render
