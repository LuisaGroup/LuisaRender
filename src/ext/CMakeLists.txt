set(MI_OSX_ZONE ON CACHE BOOL "" FORCE)
set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
add_subdirectory(mimalloc)
set_target_properties(mimalloc PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(mimalloc_config INTERFACE)
target_link_libraries(mimalloc_config INTERFACE mimalloc)

set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
add_library(spdlog_config INTERFACE)
target_link_libraries(spdlog_config INTERFACE spdlog::spdlog_header_only mimalloc_config)

add_library(jitify_config INTERFACE)
target_include_directories(jitify_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(cxxopts)
add_library(cxxopts_config INTERFACE)
target_link_libraries(cxxopts_config INTERFACE cxxopts::cxxopts mimalloc_config)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
add_subdirectory(assimp)
add_library(assimp_config INTERFACE)
target_link_libraries(assimp_config INTERFACE assimp::assimp mimalloc_config)

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
    target_link_libraries(opencl_config INTERFACE OpenCL::OpenCL mimalloc_config)
endif ()
