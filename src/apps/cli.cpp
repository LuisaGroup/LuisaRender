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
    cxxopts::Options cli{"megakernel_path_tracing"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.allow_unrecognised_options();
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
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts;
        for (auto &&u : unknown) {
            opts.append(" ").append(u);
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

//    log_level_info();
    luisa::compute::Context context{argv[0]};

    auto options = parse_cli_options(argc, argv);
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<uint32_t>();
    auto path = options["scene"].as<std::filesystem::path>();

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
    auto scene_desc = SceneParser::parse(path);
    LUISA_INFO(
        "Parsed scene description "
        "file '{}' in {} ms.",
        path.string(), clock.toc());

    auto scene = Scene::create(context, scene_desc.get());
    auto stream = device.create_stream();
    auto pipeline = Pipeline::create(device, stream, *scene);
    pipeline->render(stream);
    stream.synchronize();
}
