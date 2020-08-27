//
// Created by Mike Smith on 2020/5/15.
//

#pragma once

#include "logging.h"
#include <filesystem>

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))

#include <dlfcn.h>

#define LUISA_EXPORT [[gnu::visibility("default")]]
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

inline void destroy_dynamic_module(DynamicModuleHandle handle) {
    if (handle != nullptr) { dlclose(handle); }
}

template<typename F>
inline auto load_dynamic_symbol(DynamicModuleHandle handle, const std::string &name) {
    LUISA_EXCEPTION_IF(name.empty(), "Empty name given for dynamic symbol");
    LUISA_INFO("Loading dynamic symbol: ", name);
    auto symbol = dlsym(handle, name.c_str());
    LUISA_EXCEPTION_IF(symbol == nullptr, "Failed to load dynamic symbol \"", name, "\", reason: ", dlerror());
    return reinterpret_cast<F *>(symbol);
}

}}// namespace luisa::utility

#elif defined(_WIN32) || defined(_WIN64)

#include <windowsx.h>

#define LUISA_EXPORT __declspec(dllexport)
#define LUISA_DLL_PREFIX ""
#define LUISA_DLL_EXTENSION ".dll"

namespace luisa { inline namespace utility {

using DynamicModuleHandle = HMODULE;

namespace detail {

inline std::string win32_last_error_message() {
    // Retrieve the system error message for the last-error code
    void *buffer = nullptr;
    auto err_code = GetLastError();
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        err_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&buffer,
        0, nullptr);

    auto err_msg = serialize(buffer, " (code = 0x", std::hex, err_code, ").");
    LocalFree(buffer);
    
    return err_msg;
}

}// namespace detail

inline DynamicModuleHandle load_dynamic_module(const std::filesystem::path &path) {
    LUISA_EXCEPTION_IF_NOT(std::filesystem::exists(path), "Dynamic module not found: ", path);
    LUISA_INFO("Loading dynamic module: ", path);
    auto module = LoadLibraryA(std::filesystem::canonical(path).string().c_str());
    LUISA_EXCEPTION_IF(module == nullptr, "Failed to load dynamic module ", path, ", reason: ", detail::win32_last_error_message());
    return module;
}

inline void destroy_dynamic_module(DynamicModuleHandle handle) {
    if (handle != nullptr) { FreeLibrary(handle); }
}

template<typename F>
inline auto load_dynamic_symbol(DynamicModuleHandle handle, const std::string &name) {
    LUISA_EXCEPTION_IF(name.empty(), "Empty name given for dynamic symbol");
    LUISA_INFO("Loading dynamic symbol: ", name);
    auto symbol = GetProcAddress(handle, name.c_str());
    LUISA_EXCEPTION_IF(symbol == nullptr, "Failed to load dynamic symbol \"", name, "\", reason: ", detail::win32_last_error_message());
    return reinterpret_cast<F *>(symbol);
}

}}// namespace luisa::utility

#else
#error Unsupported platform for DLL exporting and importing
#endif
