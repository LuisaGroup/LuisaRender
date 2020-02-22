//
// Created by Mike Smith on 2020/2/3.
//

#pragma once

#include <exception>
#include <iostream>
#include "string_manipulation.h"

#define LUISA_WARNING(...)  \
    [&] { std::cerr << "[WARNING] " << __FILE__ << ":" << __LINE__ << ": " << serialize(__VA_ARGS__) << std::endl; }()
    
#define LUISA_ERROR(...)  \
    throw std::runtime_error{serialize("[ERROR] ", __FILE__, ":", __LINE__, ": ", __VA_ARGS__)}

#define LUISA_WARNING_IF(condition, ...)      [&] { if (condition) { LUISA_WARNING(__VA_ARGS__); } }()
#define LUISA_WARNING_IF_NOT(condition, ...)  LUISA_WARNING_IF(!(condition), __VA_ARGS__)

#define LUISA_ERROR_IF(condition, ...)        [&] { if (condition) { LUISA_ERROR(__VA_ARGS__); } }()
#define LUISA_ERROR_IF_NOT(condition, ...)    LUISA_ERROR_IF(!(condition), __VA_ARGS__)
