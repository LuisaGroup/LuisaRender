//
// Created by Mike on 2021/12/7.
//

#include <span>
#include <iostream>

#include <cxxopts.hpp>

#include <core/stl/format.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"luisa-render-cli"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<int32_t>()->default_value("-1"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.add_option("", "D", "define", "Parameter definitions to override scene description macros.",
                   cxxopts::value<std::vector<luisa::string>>()->default_value("<none>"), "<key>=<value>");
    cli.add_option("", "h", "help", "Display this help message", cxxopts::value<bool>()->default_value("false"), "");
    cli.allow_unrecognised_options();
    cli.positional_help("<file>");
    cli.parse_positional("scene");
    auto options = [&] {
        try {
            return cli.parse(argc, argv);
        } catch (const std::exception &e) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to parse command line arguments: {}.",
                e.what());
            std::cout << cli.help() << std::endl;
            exit(-1);
        }
    }();
    if (options["help"].as<bool>()) {
        std::cout << cli.help() << std::endl;
        exit(0);
    }
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts{unknown.front()};
        for (auto &&u : luisa::span{unknown}.subspan(1)) {
            opts.append("; ").append(u);
        }
        LUISA_WARNING_WITH_LOCATION(
            "Unrecognized options: {}", opts);
    }
    return options;
}

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

[[nodiscard]] auto parse_cli_macros(int &argc, char *argv[]) {
    SceneParser::MacroMap macros;
    auto parse_macro = [&macros](luisa::string_view d) noexcept {
        if (auto p = d.find('='); p == luisa::string::npos) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid definition: {}", d);
        } else {
            auto key = d.substr(0, p);
            auto value = d.substr(p + 1);
            LUISA_VERBOSE_WITH_LOCATION("Parameter definition: {} = '{}'", key, value);
            if (auto iter = macros.find(key); iter != macros.end()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate definition: {} = '{}'. "
                    "Ignoring the previous one: {} = '{}'.",
                    key, value, key, iter->second);
                iter->second = value;
            } else {
                macros.emplace(key, value);
            }
        }
    };
    // parse all options starting with '-D' or '--define'
    for (int i = 1; i < argc; i++) {
        auto arg = luisa::string_view{argv[i]};
        if (arg == "-D" || arg == "--define") {
            if (i + 1 == argc) {
                LUISA_WARNING_WITH_LOCATION(
                    "Missing definition after {}.", arg);
                // remove the option
                argv[i] = nullptr;
            } else {
                parse_macro(argv[i + 1]);
                // remove the option and its argument
                argv[i] = nullptr;
                argv[++i] = nullptr;
            }
        } else if (arg.starts_with("-D")) {
            parse_macro(arg.substr(2));
            // remove the option
            argv[i] = nullptr;
        }
    }
    // remove all nullptrs
    auto new_end = std::remove(argv, argv + argc, nullptr);
    argc = static_cast<int>(new_end - argv);
    return macros;
}


