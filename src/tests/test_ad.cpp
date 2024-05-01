
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

#include <random>
#include <luisa-compute.h>
#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <sdl/scene_node_desc.h>
#include <limits>

using namespace luisa;
using namespace luisa::render;
using namespace luisa::compute;
//

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

int main(int argc, char *argv[]) {

    log_level_info();
    luisa::log_level_verbose();
    luisa::compute::Context context{argv[0]};
    auto macros = parse_cli_macros(argc, argv);
    for (auto &&[k, v] : macros) {
        LUISA_INFO("Found CLI Macro: {} = {}", k, v);
    }

    auto options = parse_cli_options(argc, argv);
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<int32_t>();
    auto path = options["scene"].as<std::filesystem::path>();
    compute::DeviceConfig config;
    config.device_index = index;
    config.inqueue_buffer_limit = false;// Do not limit the number of in-queue buffers --- we are doing offline rendering!
    auto device = context.create_device(backend, &config);

    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();

    LUISA_INFO("Parsed scene description file '{}' in {} ms.",
               path.string(), parse_time);
    auto scene = Scene::create(context, scene_desc.get());
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto pipeline = Pipeline::create(device, stream, *scene);

    Kernel1D ad_kernel = [] () {
        $autodiff {
            Float3 point_pre = make_float3(-0.021873882, 1.9799696, -0.018273553);
            Float3 point_cur = make_float3(0.5625003, 1, -0.52317667);
            Float3 point_nxt = make_float3(0.8394947, 0, -0.7626182);
            Float3 normal_cur =  make_float3(0.,1.,0.);
            device_log("point_pre {} point_cur {} point_nxt {} normal_cur {}",point_pre, point_cur, point_nxt, normal_cur);
            requires_grad(point_pre, point_cur, point_nxt, normal_cur);
            auto wi = normalize(point_pre-point_cur);
            auto wo = normalize(point_nxt-point_cur);
            auto s = normalize(make_float3(0.0f, -normal_cur[2], normal_cur[1]));
            // auto ss = normalize(s - n * dot(n, s));
            // auto tt = normalize(cross(n, ss));
            auto f = Frame::make(normal_cur, s);
            auto wi_local = f.world_to_local(wi);
            auto wo_local = f.world_to_local(wo);
            backward(wi);
            auto res = normalize(wi_local+wo_local*1.8f);   
            device_log("res is {} wi {} wo {} eta {} normal {}",res, wi_local, wo_local, 1.8, normal_cur);
            device_log("grad(normal_cur) is {}",grad(normal_cur));
            device_log("grad(point_pre) is {}",grad(point_pre));
            device_log("grad(point_cur) is {}",grad(point_cur));
            device_log("grad(point_nxt) is {}",grad(point_nxt));
        };
    };
    stream << device.compile(ad_kernel)().dispatch(1u) << synchronize();
}
