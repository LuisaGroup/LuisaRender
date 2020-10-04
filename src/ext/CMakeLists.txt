set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
add_library(spdlog_config INTERFACE)
target_link_libraries(spdlog_config INTERFACE spdlog::spdlog)

add_library(jitify_config INTERFACE)
target_include_directories(jitify_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(cxxopts)
add_library(cxxopts_config INTERFACE)
target_link_libraries(cxxopts_config INTERFACE cxxopts::cxxopts)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
add_subdirectory(assimp)
add_library(assimp_config INTERFACE)
target_link_libraries(assimp_config INTERFACE assimp::assimp)

add_library(tinyexr_config INTERFACE)
target_include_directories(tinyexr_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(stb_config INTERFACE)
target_include_directories(stb_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (LUISA_RENDER_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
    add_library(opencl_config INTERFACE)
    target_compile_definitions(opencl_config INTERFACE
                               CL_HPP_TARGET_OPENCL_VERSION=120
                               CL_HPP_MINIMUM_OPENCL_VERSION=120
                               CL_HPP_CL_1_2_DEFAULT_BUILD
                               CL_HPP_ENABLE_EXCEPTIONS)
    target_include_directories(opencl_config INTERFACE OpenCL-CLHPP/include)
    target_link_libraries(opencl_config INTERFACE OpenCL::OpenCL)
endif ()
