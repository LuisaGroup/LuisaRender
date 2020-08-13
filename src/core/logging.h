//
// Created by Mike Smith on 2020/2/3.
//

#pragma once

#include <exception>
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "string_manipulation.h"

namespace luisa { inline namespace utility {

spdlog::logger &logger() noexcept;

template<typename ...Args>
inline void LUISA_INFO(Args &&...args) noexcept {
    logger().info(serialize(std::forward<Args>(args)...));
}

template<typename ...Args>
inline void LUISA_WARNING(Args &&...args) noexcept {
    logger().warn(serialize(std::forward<Args>(args)...));
}

template<typename ...Args>
inline void LUISA_WARNING_IF(bool predicate, Args &&...args) noexcept {
    if (predicate) { LUISA_WARNING(std::forward<Args>(args)...); }
}

template<typename ...Args>
inline void LUISA_WARNING_IF_NOT(bool predicate, Args &&...args) noexcept {
    LUISA_WARNING_IF(!predicate, std::forward<Args>(args)...);
}

template<typename ...Args>
[[noreturn]] inline void LUISA_EXCEPTION(Args &&...args) {
    throw std::runtime_error{serialize(std::forward<Args>(args)...)};
}

template<typename ...Args>
inline void LUISA_EXCEPTION_IF(bool predicate, Args &&...args) {
    if (predicate) { LUISA_EXCEPTION(std::forward<Args>(args)...); }
}

template<typename ...Args>
inline void LUISA_EXCEPTION_IF_NOT(bool predicate, Args &&...args) {
    LUISA_EXCEPTION_IF(!predicate, std::forward<Args>(args)...);
}

template<typename ...Args>
[[noreturn]] inline void LUISA_ERROR(Args &&...args) {
    logger().error(serialize(std::forward<Args>(args)...));
    exit(-1);
}

template<typename ...Args>
inline void LUISA_ERROR_IF(bool predicate, Args &&...args) {
    if (predicate) { LUISA_ERROR(std::forward<Args>(args)...); }
}

template<typename ...Args>
inline void LUISA_ERROR_IF_NOT(bool predicate, Args &&...args) {
    LUISA_ERROR_IF(!predicate, std::forward<Args>(args)...);
}

}}
