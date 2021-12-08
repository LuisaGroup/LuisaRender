//
// Created by Mike on 2021/12/7.
//

#include <cxxopts.hpp>
#include <luisa-compute.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"Mega-Kernel Path Tracing"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<std::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
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
    if (options["backend"].count() == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Compute backend not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    return options;
}

int main(int argc, char *argv[]) {

    luisa::compute::Context context{argv[0]};

    auto options = parse_cli_options(argc, argv);
    auto backend = options["backend"].as<std::string>();
    auto index = options["device"].as<uint32_t>();

    auto device = context.create_device(backend, {{"index", index}});
}
