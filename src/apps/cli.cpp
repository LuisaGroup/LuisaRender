//
// Created by Mike on 2021/12/7.
//

#include <span>

#include <cxxopts.hpp>

#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>

#include <util/ies.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"luisa-render-cli"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.add_option("", "D", "define", "Parameter definitions to override scene description macros.",
                   cxxopts::value<std::vector<std::string>>()->default_value("<none>"), "<key>=<value>");
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

int main(int argc, char *argv[]) {

    log_level_info();
    auto options = parse_cli_options(argc, argv);

    luisa::compute::Context context{argv[0]};
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<uint32_t>();
    auto path = options["scene"].as<std::filesystem::path>();
    auto definitions = options["define"].as<std::vector<std::string>>();
    SceneParser::MacroMap macros;
    for (luisa::string_view d : definitions) {
        if (d == "<none>") { continue; }
        auto p = d.find('=');
        if (p == luisa::string::npos) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid definition: {}", d);
            continue;
        }
        auto key = d.substr(0, p);
        auto value = d.substr(p + 1);
        LUISA_INFO("Parameter definition: {} = '{}'", key, value);
        if (auto iter = macros.find(key); iter != macros.end()) {
            LUISA_WARNING_WITH_LOCATION(
                "Duplicate definition: {} = '{}'. "
                "Ignoring the previous one: {} = '{}'.",
                key, value, key, iter->second);
        }
        macros[key] = value;
    }

    //    auto ies_profile = IESProfile::parse("/Users/mike/Downloads/002bb0e37aa7e5f1d7851fb1db032628.ies");
    //    LUISA_INFO(
    //        "Loaded IES profile with "
    //        "{} vertical angle(s) and "
    //        "{} horizontal angle(s):",
    //        ies_profile.vertical_angles().size(),
    //        ies_profile.horizontal_angles().size());
    //    std::cout << "Vertical Angles:";
    //    for (auto v : ies_profile.vertical_angles()) {
    //        std::cout << " " << v;
    //    }
    //    std::cout << "\n";
    //    auto candela_offset = 0u;
    //    for (auto h : ies_profile.horizontal_angles()) {
    //        std::cout << "@" << h << ":";
    //        for (auto i = 0u; i < ies_profile.vertical_angles().size(); i++) {
    //            std::cout << " " << ies_profile.candela_values()[candela_offset + i];
    //        }
    //        std::cout << "\n";
    //        candela_offset += ies_profile.vertical_angles().size();
    //    }

    auto device = context.create_device(
        backend, luisa::format(R"({{"index": {}}})", index));
    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();

    LUISA_INFO("Parsed scene description file '{}' in {} ms.",
               path.string(), parse_time);
    auto scene = Scene::create(context, scene_desc.get());
    auto stream = device.create_stream(false);
    auto pipeline = Pipeline::create(device, stream, *scene);
    pipeline->render(stream);
    stream.synchronize();
}
