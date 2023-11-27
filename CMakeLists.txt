cmake_minimum_required(VERSION 3.18...3.21)
cmake_policy(VERSION 3.18)

cmake_policy(SET CMP0069 NEW)
cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0091 NEW)

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(LuisaRender LANGUAGES C CXX VERSION 0.1)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "LuisaRender only supports 64-bit platforms")
endif ()

if (NOT DEFINED LUISA_RENDER_MASTER_PROJECT)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(LUISA_RENDER_MASTER_PROJECT ON)
    else ()
        set(LUISA_RENDER_MASTER_PROJECT OFF)
    endif ()
endif ()

option(LUISA_RENDER_BUILD_TESTS "Build tests for LuisaRender" ${LUISA_RENDER_MASTER_PROJECT})
option(LUISA_RENDER_ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" OFF)

include(src/compute/scripts/setup_compilation.cmake)

add_subdirectory(src)
