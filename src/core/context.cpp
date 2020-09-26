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
std::filesystem::path Context::cache_path(const std::filesystem::path &name) noexcept { return _runtime_dir() / "cache" / name; }
std::filesystem::path Context::input_path(const std::filesystem::path &name) noexcept { return _input_dir() / name; }

Context::~Context() noexcept {
    for (auto &&module_item : _loaded_modules) { destroy_dynamic_module(module_item.second); }
}

Context::Context(int argc, char *argv[])
    : _argc{argc},
      _argv{const_cast<const char **>(argv)},
      _cli_options{std::filesystem::path{argv[0]}.filename().string()} {
    
    _cli_options.add_options()
                    ("d,devices", "Select compute devices", cxxopts::value<std::vector<std::string>>()->default_value(""))
                    ("runtime-dir",
                     "Specify runtime directory",
                     cxxopts::value<std::filesystem::path>()->default_value(std::filesystem::canonical(argv[0]).parent_path().parent_path().string()))
                    ("working-dir",
                     "Specify working directory",
                     cxxopts::value<std::filesystem::path>()->default_value(std::filesystem::canonical(std::filesystem::current_path()).string()))
                    ("clear-cache", "Clear cached kernel compilation", cxxopts::value<bool>())
                    ("print-source", "Print generated source code", cxxopts::value<bool>())
                    ("positional", "Specify input file", cxxopts::value<std::string>());
}

const cxxopts::ParseResult &Context::_parse_result() const noexcept {
    if (!_parsed_cli_options.has_value()) {
        _cli_options.parse_positional("positional");
        _parsed_cli_options.emplace(_cli_options.parse(const_cast<int &>(_argc), const_cast<const char **&>(_argv)));
    }
    return *_parsed_cli_options;
}

const std::filesystem::path &Context::_runtime_dir() noexcept {
    if (_run_dir.empty()) {
        _run_dir = std::filesystem::canonical(_parse_result()["runtime-dir"].as<std::filesystem::path>());
        LUISA_EXCEPTION_IF(!std::filesystem::exists(_run_dir) || !std::filesystem::is_directory(_run_dir), "Invalid runtime directory: ", _run_dir);
        LUISA_INFO("Runtime directory: ", _run_dir);
        auto cache_directory = _run_dir / "cache";
        if (_parse_result()["clear-cache"].as<bool>() && std::filesystem::exists(cache_directory)) {
            LUISA_INFO("Removing cache directory: ", cache_directory);
            std::filesystem::remove_all(cache_directory);
        }
        LUISA_EXCEPTION_IF(!_create_folder_if_necessary(cache_directory), "Failed to create cache directory: ", cache_directory);
    }
    return _run_dir;
}

const std::filesystem::path &Context::_working_dir() noexcept {
    if (_work_dir.empty()) {
        _work_dir = std::filesystem::canonical(_parse_result()["working-dir"].as<std::filesystem::path>());
        LUISA_EXCEPTION_IF(!std::filesystem::exists(_work_dir) || !std::filesystem::is_directory(_work_dir), "Invalid working directory: ", _work_dir);
        std::filesystem::current_path(_work_dir);
        LUISA_INFO("Working directory: ", _work_dir);
    }
    return _work_dir;
}

const std::vector<Context::DeviceSelection> &Context::device_selections() noexcept {
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

const std::filesystem::path &Context::_input_dir() noexcept {
    if (_in_dir.empty()) {
        if (_parse_result().count("positional") == 0u) {
            LUISA_WARNING("No positional CLI argument given, setting input directory to working directory: ", _working_dir());
        } else {
            _in_dir = std::filesystem::canonical(cli_positional_option()).parent_path();
            LUISA_EXCEPTION_IF(!std::filesystem::exists(_in_dir) || !std::filesystem::is_directory(_in_dir), "Invalid input directory: ", _in_dir);
            LUISA_INFO("Input directory: ", _in_dir);
        }
    }
    return _in_dir;
}

}// namespace luisa
