//
// Created by Mike Smith on 2020/5/15.
//

#pragma once

#include <cstdlib>
#include <filesystem>

#include "logging.h"

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))

#include <dlfcn.h>
#include <unistd.h>

#define LUISA_EXPORT [[gnu::visibility("default")]]
#define LUISA_DLL_PREFIX "lib"
#define LUISA_DLL_EXTENSION ".so"

namespace luisa { inline namespace utility {

inline size_t memory_page_size() noexcept {
    static thread_local auto page_size = getpagesize();
    return page_size;
}

inline void *aligned_alloc(size_t alignment, size_t size) noexcept {
    return ::aligned_alloc(alignment, size);
}

inline void aligned_free(void *buffer) noexcept {
    free(buffer);
}

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

inline size_t memory_page_size() noexcept {
    static thread_local auto page_size = [] {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return info.dwPageSize;
    }();
    return page_size;
}

inline void *aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}

inline void aligned_free(void *buffer) noexcept {
    _aligned_free(buffer);
}

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

namespace luisa {

inline namespace utility {

template<typename T>
class PageAlignedMemory {

public:
    inline static auto page_size = memory_page_size();

private:
    T *_memory;
    size_t _aligned_byte_size;

public:
    explicit PageAlignedMemory(size_t size) noexcept {
        _aligned_byte_size = std::max((size * sizeof(T) + page_size - 1u) / page_size * page_size, page_size);
        _memory = reinterpret_cast<T *>(luisa::aligned_alloc(page_size, _aligned_byte_size));
    }

    PageAlignedMemory(PageAlignedMemory &&another) noexcept
        : _memory{another._memory}, _aligned_byte_size{another._aligned_byte_size} {
        another._memory = nullptr;
        another._aligned_byte_size = 0ul;
    }

    PageAlignedMemory &operator=(PageAlignedMemory &&rhs) noexcept {
        _memory = rhs._memory;
        _aligned_byte_size = rhs._aligned_byte_size;
        rhs._memory = nullptr;
        rhs._aligned_byte_size = 0ul;
        return *this;
    }

    PageAlignedMemory(const PageAlignedMemory &) = delete;
    PageAlignedMemory &operator=(const PageAlignedMemory &) = delete;

    ~PageAlignedMemory() noexcept {
        if (_memory != nullptr) { luisa::aligned_free(_memory); }
    }

    [[nodiscard]] size_t aligned_byte_size() const noexcept { return _aligned_byte_size; }
    [[nodiscard]] T *data() noexcept { return _memory; }
    [[nodiscard]] const T *data() const noexcept { return _memory; }
};

}}// namespace luisa::utility
