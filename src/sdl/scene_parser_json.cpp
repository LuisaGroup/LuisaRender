//
// Created by Mike Smith on 2022/9/26.
//

#include <fstream>
#include <streambuf>

#include <nlohmann/json.hpp>
#include <util/thread_pool.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_node_desc.h>
#include <sdl/scene_parser.h>
#include <sdl/scene_parser_json.h>

namespace luisa::render {

SceneParserJSON::SceneParserJSON(SceneDesc &desc, const std::filesystem::path &path,
                                 const MacroMap &cli_macros) noexcept
    : _desc{desc}, _cli_macros{cli_macros},
      _location{desc.register_path(std::filesystem::canonical(path))} {}


// SceneParserJSON::SceneParserJSON(SceneDesc &desc, std::string json_source,
//                                  const MacroMap &cli_macros) noexcept
//     : _desc{desc}, _cli_macros{cli_macros},
//       json_source{json_source} {}

void SceneParserJSON::parse() const noexcept {
    auto root = [this] {
        std::ifstream ifs{*_location.file()};
        std::string src{std::istreambuf_iterator<char>{ifs},
                        std::istreambuf_iterator<char>{}};
        return json::parse(src, nullptr, false, true);
    }();
    _parse_root(root);
}

// void SceneParserJSON::parse() const noexcept {
//     auto root = [this] {
//         if(_location.file()== nullptr) {
//             return json::parse(json_source, nullptr, false, true);
//         }
//         std::ifstream ifs{*_location.file()};
//         std::string src{std::istreambuf_iterator<char>{ifs},
//                         std::istreambuf_iterator<char>{}};
//         return json::parse(src, nullptr, false, true);
//     }();
//     _parse_root(root);
// }


void SceneParserJSON::_parse_node(SceneNodeDesc &desc, const json &node) const noexcept {

    auto parse_internal = [&desc, &node, this](luisa::string_view key, const json &n) noexcept {
        LUISA_ASSERT(n.is_object(), "Invalid node reference in '{}'.'{}': {}",
                     desc.identifier(), key, node.dump(2));
        for (auto &&prop : n.items()) {
            LUISA_ASSERT(prop.key() == "type" || prop.key() == "impl" ||
                             prop.key() == "base" || prop.key() == "prop",
                         "Invalid internal node property '{}.{}': {}",
                         key, prop.key(), prop.value().dump(2));
        }
        auto n_impl_desc = n.at("impl").get<luisa::string>();
        const SceneNodeDesc *n_base = nullptr;
        if (auto iter = n.find("base"); iter != n.end()) {
            n_base = _reference(iter->get<luisa::string>());
        }
        auto internal = desc.define_internal(n_impl_desc, _location, n_base);
        if (auto iter_prop = n.find("prop"); iter_prop != n.end()) {
            _parse_node(*internal, iter_prop.value());
        }
        return internal;
    };

    for (auto &&item : node.items()) {
        // process properties
        if (item.value().is_string()) {
            auto value = item.value().get<luisa::string>();
            if (value.starts_with('@')) {
                desc.add_property(item.key(), _reference(value));
            } else {
                desc.add_property(item.key(), std::move(value));
            }
        } else if (item.value().is_number()) {
            desc.add_property(item.key(), item.value().get<double>());
        } else if (item.value().is_boolean()) {
            desc.add_property(item.key(), item.value().get<bool>());
        } else if (item.value().is_array()) {
            auto &&array = item.value();
            LUISA_ASSERT(!array.empty(), "Empty array is not allowed in '{}'.'{}': {}",
                         desc.identifier(), item.key(), node.dump(2));
            if (array[0].is_string()) {
                auto s = array[0].get<luisa::string>();
                if (s.starts_with('@')) {// node array
                    luisa::vector<const SceneNodeDesc *> nodes;
                    nodes.reserve(array.size());
                    nodes.emplace_back(_reference(s));
                    for (auto i = 1u; i < array.size(); i++) {
                        auto &&n = array[i];
                        if (n.is_string()) {// reference
                            nodes.emplace_back(_reference(n.get<luisa::string>()));
                        } else {// internal node
                            nodes.emplace_back(parse_internal(item.key(), n));
                        }
                    }
                    desc.add_property(item.key(), std::move(nodes));
                } else {// string array
                    luisa::vector<luisa::string> values;
                    values.reserve(array.size());
                    values.emplace_back(std::move(s));
                    for (auto i = 1u; i < array.size(); i++) {
                        values.emplace_back(array[i].get<luisa::string>());
                    }
                    desc.add_property(item.key(), std::move(values));
                }
            } else if (array[0].is_number()) {
                luisa::vector<double> values;
                values.reserve(array.size());
                for (auto &&v : array) { values.emplace_back(v.get<double>()); }
                desc.add_property(item.key(), std::move(values));
            } else if (array[0].is_boolean()) {
                luisa::vector<bool> values;
                values.reserve(array.size());
                for (auto &&v : array) { values.emplace_back(v.get<bool>()); }
                desc.add_property(item.key(), std::move(values));
            } else {
                luisa::vector<const SceneNodeDesc *> nodes;
                nodes.reserve(array.size());
                for (auto &&n : array) {
                    if (n.is_string()) {// reference
                        nodes.emplace_back(_reference(n.get<luisa::string>()));
                    } else {// internal node
                        nodes.emplace_back(parse_internal(item.key(), n));
                    }
                }
                desc.add_property(item.key(), std::move(nodes));
            }
        } else {// inline nodes
            if (!item.value().is_null()) {
                desc.add_property(item.key(), parse_internal(item.key(), item.value()));
            }
        }
    }
}

void SceneParserJSON::_parse_import(const json &node) const noexcept {
    auto dispatch_parse = [this](auto file_name) {
        std::filesystem::path path{file_name};
        if (!path.is_absolute()) { path = _location.file()->parent_path() / path; }
        global_thread_pool().async([path = std::move(path), &desc = _desc, &cli_macros = _cli_macros] {
            SceneParser::_dispatch_parse(desc, path, cli_macros);
        });
    };
    if (node.is_string()) {
        dispatch_parse(node.get<luisa::string>());
    } else if (node.is_array()) {
        for (auto &&file_name : node) {
            dispatch_parse(file_name.get<luisa::string>());
        }
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid import node:\n{}",
            node.dump(2));
    }
}

const SceneNodeDesc *SceneParserJSON::_reference(luisa::string_view name) const noexcept {
    LUISA_ASSERT(name.starts_with('@'), "Invalid reference name '{}'.", name);
    return _desc.reference(name.substr(1));
}

void SceneParserJSON::_parse_root(const json &root) const noexcept {
    // process imports first to fully utilize the thread pool
    if (auto iter = root.find("import"); iter != root.end()) {
        _parse_import(iter.value());
    }
    for (auto &&item : root.items()) {
        if (item.key() == SceneDesc::root_node_identifier) {// render node
            LUISA_ASSERT(item.value().is_object(),
                         "Invalid render node: {}",
                         item.value().dump(2));
            auto render = _desc.define_root(_location);
            _parse_node(*render, item.value());
        } else if (item.key() != "import") {// global node
            LUISA_ASSERT(item.value().is_object(),
                         "Invalid global node '{}': {}",
                         item.key(), item.value().dump(2));
            for (auto &&prop : item.value().items()) {
                LUISA_ASSERT(prop.key() == "type" || prop.key() == "impl" ||
                                 prop.key() == "base" || prop.key() == "prop",
                             "Invalid global node property '{}.{}': {}",
                             item.key(), prop.key(), prop.value().dump(2));
            }
            LUISA_ASSERT(item.value().contains("type"),
                         "Missing node type in global node '{}': {}",
                         item.key(), item.value().dump(2));
            auto type_desc = item.value().at("type").get<luisa::string>();
            auto tag = parse_scene_node_tag(type_desc);
            if (tag == SceneNodeTag::ROOT) {
                LUISA_ERROR_WITH_LOCATION(
                    "Unknown scene node type: {}\n{}: {}",
                    type_desc, item.key(), item.value().dump(2));
            }
            auto impl_desc = item.value().at("impl").get<luisa::string>();
            const SceneNodeDesc *base = nullptr;
            if (auto iter = item.value().find("base"); iter != item.value().end()) {
                LUISA_ASSERT(iter->is_string(), "Invalid base node: {}", iter->dump(2));
                base = _reference(iter->get<luisa::string>());
            }
            auto global = _desc.define(item.key(), tag, impl_desc, _location, base);
            if (auto iter_prop = item.value().find("prop"); iter_prop != item.value().end()) {
                _parse_node(*global, iter_prop.value());
            }
        }
    }
}

}// namespace luisa::render
