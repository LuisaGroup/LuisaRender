//
// Created by Mike Smith on 2020/5/15.
//

#pragma once

#include <filesystem>
#include <map>

#include <core/concepts.h>
#include <core/dll.h>

namespace luisa {

class Context : Noncopyable {

private:
    std::filesystem::path _runtime_directory;
    std::filesystem::path _working_directory;
    mutable std::map<std::filesystem::path, DynamicModuleHandle, std::less<>> _loaded_modules;
    
    static bool _create_folder_if_necessary(const std::filesystem::path &path) noexcept;

public:
    Context(int argc, const char *const argv[]);
    Context(const std::filesystem::path &runtime_dir, const std::filesystem::path &working_dir);
    ~Context() noexcept;
    bool create_working_folder(const std::filesystem::path &name) const noexcept;
    bool create_cache_folder(const std::filesystem::path &name) const noexcept;
    [[nodiscard]] std::filesystem::path include_path(const std::filesystem::path &name = {}) const noexcept;
    [[nodiscard]] std::filesystem::path working_path(const std::filesystem::path &name = {}) const noexcept;
    [[nodiscard]] std::filesystem::path runtime_path(const std::filesystem::path &name = {}) const noexcept;
    [[nodiscard]] std::filesystem::path cache_path(const std::filesystem::path &name = {}) const noexcept;
    
    template<typename F>
    [[nodiscard]] auto load_dynamic_function(const std::filesystem::path &path, std::string_view module, std::string_view function) const {
        LUISA_EXCEPTION_IF(module.empty(), "Empty name given for dynamic module");
        auto module_path = std::filesystem::canonical(path / serialize(LUISA_DLL_PREFIX, module, LUISA_DLL_EXTENSION));
        auto iter = _loaded_modules.find(module_path);
        if (iter == _loaded_modules.cend()) { iter = _loaded_modules.emplace(module_path, load_dynamic_module(module_path)).first; }
        return load_dynamic_symbol<F>(iter->second, std::string{function});
    }
};

}
