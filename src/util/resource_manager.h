//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#include <filesystem>
#include <functional>

#include "concepts.h"

namespace luisa { inline namespace utility {

class RAII : Noncopyable {

private:
    std::function<void()> _on_destroy;

public:
    template<typename F>
    explicit RAII(F &&f) noexcept : _on_destroy{std::forward<F>(f)} {}
    
    ~RAII() noexcept { _on_destroy(); }
};

class ResourceManager : Noncopyable {

private:
    std::filesystem::path _binary_directory;
    std::filesystem::path _working_directory;

public:
    [[nodiscard]] static ResourceManager &instance() noexcept {
        static ResourceManager manager;
        return manager;
    }
    
    void set_binary_directory(std::filesystem::path directory) noexcept {
        _binary_directory = std::move(directory);
    }
    
    void set_working_directory(std::filesystem::path directory) noexcept {
        _working_directory = std::move(directory);
    }
    
    [[nodiscard]] std::filesystem::path binary_path(std::string_view file_name) const noexcept {
        return _binary_directory / file_name;
    }
    
    [[nodiscard]] std::filesystem::path working_path(std::string_view file_name) const noexcept {
        return _working_directory / file_name;
    }
    
};

}}
