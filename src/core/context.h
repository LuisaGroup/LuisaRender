//
// Created by Mike Smith on 2020/5/15.
//

#pragma once

#include <filesystem>
#include <map>

#include <cxxopts.hpp>

#include <core/concepts.h>
#include <core/platform.h>

namespace luisa {

class Context : Noncopyable {

public:
    struct DeviceSelection {
        std::string backend_name;
        uint32_t device_id;
        DeviceSelection(std::string_view backend, uint32_t index) noexcept : backend_name{backend}, device_id{index} {}
    };

private:
    int _argc;
    const char **_argv;
    std::filesystem::path _run_dir;
    std::filesystem::path _work_dir;
    std::filesystem::path _in_dir;
    std::optional<std::vector<DeviceSelection>> _devices;
    mutable cxxopts::Options _cli_options;
    mutable std::optional<cxxopts::ParseResult> _parsed_cli_options;
    mutable std::optional<std::string> _positional_option;
    std::map<std::filesystem::path, DynamicModuleHandle, std::less<>> _loaded_modules;

    [[nodiscard]] const cxxopts::ParseResult &_parse_result() const noexcept;
    [[nodiscard]] const std::filesystem::path &_runtime_dir() noexcept;
    [[nodiscard]] const std::filesystem::path &_working_dir() noexcept;
    [[nodiscard]] const std::filesystem::path &_input_dir() noexcept;

    static bool _create_folder_if_necessary(const std::filesystem::path &path) noexcept;

public:
    Context(int argc, char *argv[]);
    ~Context() noexcept;
    bool create_working_folder(const std::filesystem::path &name) noexcept;
    bool create_cache_folder(const std::filesystem::path &name) noexcept;
    [[nodiscard]] std::filesystem::path include_path(const std::filesystem::path &name = {}) noexcept;
    [[nodiscard]] std::filesystem::path working_path(const std::filesystem::path &name = {}) noexcept;
    [[nodiscard]] std::filesystem::path runtime_path(const std::filesystem::path &name = {}) noexcept;
    [[nodiscard]] std::filesystem::path input_path(const std::filesystem::path &name = {}) noexcept;
    [[nodiscard]] std::filesystem::path cache_path(const std::filesystem::path &name = {}) noexcept;

    template<typename F>
    [[nodiscard]] auto load_dynamic_function(const std::filesystem::path &path, std::string_view module, std::string_view function) {
        LUISA_EXCEPTION_IF(module.empty(), "Empty name given for dynamic module");
        auto module_path = std::filesystem::canonical(path / serialize(LUISA_DLL_PREFIX, module, LUISA_DLL_EXTENSION));
        auto iter = _loaded_modules.find(module_path);
        if (iter == _loaded_modules.cend()) { iter = _loaded_modules.emplace(module_path, load_dynamic_module(module_path)).first; }
        return load_dynamic_symbol<F>(iter->second, std::string{function});
    }

    template<typename T>
    void add_cli_option(const std::string &opt_name, const std::string &desc, const std::string &default_val = {}, const std::string &implicit_val = {}) {
        if (!default_val.empty() && !implicit_val.empty()) {
            _cli_options.add_options()(opt_name, desc, cxxopts::value<T>()->default_value(default_val)->implicit_value(implicit_val));
        } else if (!default_val.empty()) {
            _cli_options.add_options()(opt_name, desc, cxxopts::value<T>()->default_value(default_val));
        } else if (!implicit_val.empty()) {
            _cli_options.add_options()(opt_name, desc, cxxopts::value<T>()->implicit_value(implicit_val));
        } else {
            _cli_options.add_options()(opt_name, desc, cxxopts::value<T>());
        }
    }

    template<typename T>
    [[nodiscard]] T cli_option(const std::string &opt_name) const { return _parse_result()[opt_name].as<T>(); }
    
    [[nodiscard]] std::string cli_positional_option() const { return _parse_result()["positional"].as<std::string>(); }

    [[nodiscard]] const std::vector<DeviceSelection> &device_selections() noexcept;
    [[nodiscard]] bool should_print_generated_source() const noexcept { return cli_option<bool>("print-source"); }
};

}// namespace luisa
