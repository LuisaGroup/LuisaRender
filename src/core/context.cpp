//
// Created by Mike Smith on 2020/5/15.
//

#include "context.h"

namespace luisa {

Context::Context(const std::filesystem::path &runtime_dir, const std::filesystem::path &working_dir)
    : _runtime_directory{std::filesystem::canonical(runtime_dir)},
      _working_directory{std::filesystem::canonical(working_dir)} {
    
    LUISA_EXCEPTION_IF(!std::filesystem::exists(runtime_dir) || !std::filesystem::is_directory(runtime_dir), "Invalid runtime directory: ", runtime_dir);
    LUISA_EXCEPTION_IF(!std::filesystem::exists(working_dir) || !std::filesystem::is_directory(working_dir), "Invalid working directory: ", working_dir);
    
    LUISA_INFO("Runtime directory: ", _runtime_directory);
    LUISA_INFO("Working directory: ", _working_directory);
    
    auto cache_directory = _working_directory / "cache";
    LUISA_EXCEPTION_IF(!_create_folder_if_necessary(cache_directory), "Failed to create cache directory: ", cache_directory);
}

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

bool Context::create_working_folder(const std::filesystem::path &name) const noexcept { return _create_folder_if_necessary(working_path(name)); }
bool Context::create_cache_folder(const std::filesystem::path &name) const noexcept { return _create_folder_if_necessary(cache_path(name)); }
std::filesystem::path Context::include_path(const std::filesystem::path &name) const noexcept { return _runtime_directory / "include" / name; }
std::filesystem::path Context::working_path(const std::filesystem::path &name) const noexcept { return _working_directory / name; }
std::filesystem::path Context::runtime_path(const std::filesystem::path &name) const noexcept { return _runtime_directory / name; }
std::filesystem::path Context::cache_path(const std::filesystem::path &name) const noexcept { return _runtime_directory / "cache" / name; }

Context::~Context() noexcept { for (auto &&module_item : _loaded_modules) { destroy_dynamic_module(module_item.second); }}

Context::Context(int argc, const char *const argv[])
    : Context{std::filesystem::canonical(argv[0]).parent_path().parent_path(),
              std::filesystem::canonical(std::filesystem::current_path())} {}
    
}
