//
// Created by Mike Smith on 2020/5/15.
//

#pragma once

#include <filesystem>
#include "logging.h"

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))

#include <dlfcn.h>

#define LUISA_DLL_EXPORT extern "C" [[gnu::visibility("default")]]
#define LUISA_DLL_PREFIX "lib"
#define LUISA_DLL_EXTENSION ".so"

namespace luisa { inline namespace utility {

using DynamicModuleHandle = void *;

inline DynamicModuleHandle load_dynamic_module(const std::filesystem::path &path) {
    LUISA_EXCEPTION_IF_NOT(std::filesystem::exists(path), "Dynamic module not found: ", path);
    LUISA_INFO("Loading dynamic module: ", path);
    auto module = dlopen(std::filesystem::canonical(path).string().c_str(), RTLD_LAZY);
    LUISA_EXCEPTION_IF(module == nullptr, "Failed to load dynamic module ", path, ", reason: ", dlerror());
    return module;
}

inline void destroy_dynamic_module(DynamicModuleHandle handle) { if (handle != nullptr) { dlclose(handle); }}

template<typename F>
inline auto load_dynamic_symbol(DynamicModuleHandle handle, const std::string &name) {
    LUISA_EXCEPTION_IF(name.empty(), "Empty name given for dynamic symbol");
    LUISA_INFO("Loading dynamic symbol: ", name);
    auto symbol = dlsym(handle, name.c_str());
    LUISA_EXCEPTION_IF(symbol == nullptr, "Failed to load dynamic symbol \"", name, "\", reason: ", dlerror());
    return reinterpret_cast<F *>(symbol);
}

}}

#elif defined(_WIN32) || defined(_WIN64)

// TODO: DLL handling on Windows

#else
#error Unsupported platform for DLL exporting and importing
#endif
