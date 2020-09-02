//
// Created by Mike Smith on 2020/2/3.
//

#pragma once

#include <exception>
#include <iostream>
#include <filesystem>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "string_util.h"

namespace luisa::logging {

spdlog::logger &logger() noexcept;

template<typename... Args>
inline void info(Args &&... args) noexcept {
    logger().info(serialize(std::forward<Args>(args)...));
}

template<typename... Args>
inline void warning(Args &&... args) noexcept {
    logger().warn(serialize(std::forward<Args>(args)...));
}

template<typename... Args>
inline void warning_if(bool predicate, Args &&... args) noexcept {
    if (predicate) { warning(std::forward<Args>(args)...); }
}

template<typename... Args>
inline void warning_if_not(bool predicate, Args &&... args) noexcept {
    warning_if(!predicate, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void exception(Args &&... args) {
    throw std::runtime_error{serialize(std::forward<Args>(args)...)};
}

template<typename... Args>
inline void exception_if(bool predicate, Args &&... args) {
    if (predicate) { exception(std::forward<Args>(args)...); }
}

template<typename... Args>
inline void exception_if_not(bool predicate, Args &&... args) {
    exception_if(!predicate, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void error(Args &&... args) {
    logger().error(serialize(std::forward<Args>(args)...));
    exit(-1);
}

template<typename... Args>
inline void error_if(bool predicate, Args &&... args) {
    if (predicate) { error(std::forward<Args>(args)...); }
}

template<typename... Args>
inline void error_if_not(bool predicate, Args &&... args) {
    error_if(!predicate, std::forward<Args>(args)...);
}

}// namespace luisa::logging

#define LUISA_INFO(...) \
    ::luisa::logging::info(__VA_ARGS__)

#define LUISA_SOURCE_LOCATION __FILE__ , ":", __LINE__

#define LUISA_WARNING(...) \
    ::luisa::logging::warning(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
#define LUISA_WARNING_IF(...) \
    ::luisa::logging::warning_if(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
#define LUISA_WARNING_IF_NOT(...) \
    ::luisa::logging::warning_if_not(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)

#define LUISA_EXCEPTION(...) \
    ::luisa::logging::exception(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
#define LUISA_EXCEPTION_IF(...) \
    ::luisa::logging::exception_if(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
#define LUISA_EXCEPTION_IF_NOT(...) \
    ::luisa::logging::exception_if_not(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)

#define LUISA_ERROR(...) \
    ::luisa::logging::error(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
#define LUISA_ERROR_IF(...) \
    ::luisa::logging::error_if(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
#define LUISA_ERROR_IF_NOT(...) \
    ::luisa::logging::error_if_not(__VA_ARGS__, "\n    Source: ", LUISA_SOURCE_LOCATION)
