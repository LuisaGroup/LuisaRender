//
// Created by Mike Smith on 2020/5/15.
//

#include "context.h"

namespace luisa {

bool Context::_create_folder_if_necessary(const std::filesystem::path &path) noexcept {
    if (std::filesystem::exists(path)) { return true; }
    try {
        LUISA_INFO("Creating folder: ", path);
        return std::filesystem::create_directories(path);
    } catch (const std::filesystem::filesystem_error &e) {
        LUISA_WARNING("Failed to create folder ", path, ", reason: ", e.what());
    }
    return false;
}

bool Context::create_working_folder(const std::filesystem::path &name) noexcept { return _create_folder_if_necessary(working_path(name)); }
bool Context::create_cache_folder(const std::filesystem::path &name) noexcept { return _create_folder_if_necessary(cache_path(name)); }
std::filesystem::path Context::include_path(const std::filesystem::path &name) noexcept { return _runtime_dir() / "include" / name; }
std::filesystem::path Context::working_path(const std::filesystem::path &name) noexcept { return _working_dir() / name; }
std::filesystem::path Context::runtime_path(const std::filesystem::path &name) noexcept { return _runtime_dir() / name; }
std::filesystem::path Context::cache_path(const std::filesystem::path &name) noexcept { return _working_dir() / "cache" / name; }

Context::~Context() noexcept {
    for (auto &&module_item : _loaded_modules) { destroy_dynamic_module(module_item.second); }
}

Context::Context(int argc, char *argv[])
    : _argc{argc},
      _argv{const_cast<const char **>(argv)},
      _cli_options{std::filesystem::path{argv[0]}.filename().string()} {

    _cli_options.add_options()("d,devices", "Select compute devices", cxxopts::value<std::vector<std::string>>()->default_value(""))
                    ("rundir", "Specify runtime directory", cxxopts::value<std::filesystem::path>()->default_value(std::filesystem::canonical(argv[0]).parent_path().parent_path().string()))
                    ("workdir", "Specify working directory", cxxopts::value<std::filesystem::path>()->default_value(std::filesystem::canonical(std::filesystem::current_path()).string()))
                    ("C,clearcache", "Clear cached kernel compilation", cxxopts::value<bool>());
}

const cxxopts::ParseResult &Context::_parse_result() noexcept {
    if (!_parsed_cli_options.has_value()) { _parsed_cli_options.emplace(_cli_options.parse(_argc, _argv)); }
    return *_parsed_cli_options;
}

const std::filesystem::path &Context::_runtime_dir() noexcept {
    if (_rundir.empty()) {
        _rundir = std::filesystem::canonical(_parse_result()["rundir"].as<std::filesystem::path>());
        LUISA_EXCEPTION_IF(!std::filesystem::exists(_rundir) || !std::filesystem::is_directory(_rundir), "Invalid runtime directory: ", _rundir);
        LUISA_INFO("Runtime directory: ", _rundir);
    }
    return _rundir;
}

const std::filesystem::path &Context::_working_dir() noexcept {
    if (_workdir.empty()) {
        _workdir = std::filesystem::canonical(_parse_result()["workdir"].as<std::filesystem::path>());
        LUISA_EXCEPTION_IF(!std::filesystem::exists(_workdir) || !std::filesystem::is_directory(_workdir), "Invalid working directory: ", _workdir);
        LUISA_INFO("Working directory: ", _workdir);
        auto cache_directory = _workdir / "cache";
        if (_parse_result()["clearcache"].as<bool>() && std::filesystem::exists(cache_directory)) {
            LUISA_INFO("Removing cache directory: ", cache_directory);
            std::filesystem::remove_all(cache_directory);
        }
        LUISA_EXCEPTION_IF(!_create_folder_if_necessary(cache_directory), "Failed to create cache directory: ", cache_directory);
    }
    return _workdir;
}

const std::vector<Context::DeviceSelection> &Context::devices() noexcept {
    if (!_devices.has_value()) {
        auto &&devices = _devices.emplace();
        for (auto &&device : _parse_result()["devices"].as<std::vector<std::string>>()) {
            LUISA_INFO("Device: ", device);
            if (auto p = device.find(':'); p != std::string::npos) {
                auto index = 0u;
                std::stringstream ss;
                ss << std::string_view{device}.substr(p + 1u);
                ss >> index;
                devices.emplace_back(std::string_view{device}.substr(0u, p), index);
            } else {
                devices.emplace_back(device, 0u);
            }
        }
        if (!devices.empty()) {
            std::ostringstream ss;
            for (auto i = 0u; i < devices.size(); i++) {
                ss << devices[i].backend_name << ":" << devices[i].device_id;
                if (i != devices.size() - 1u) { ss << ", "; }
            }
            LUISA_INFO("Candidate devices: ", ss.str());
        }
    }
    return *_devices;
}

}// namespace luisa
