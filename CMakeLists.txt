cmake_minimum_required(VERSION 3.16...3.17)
cmake_policy(VERSION 3.16)

project(LuisaRender LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    set(LUISA_RENDER_ENABLE_METAL_DEFAULT ON)
else ()
    set(LUISA_RENDER_ENABLE_METAL_DEFAULT OFF)
endif ()
option(LUISA_RENDER_ENABLE_METAL "Build with Apple Metal support" ${LUISA_RENDER_ENABLE_METAL_DEFAULT})

find_package(OpenCV REQUIRED)

set(GLM_QUIET ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_CXX_17 ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/glm)
target_compile_definitions(glm INTERFACE GLM_FORCE_AVX2 GLM_FORCE_CXX17)

add_subdirectory(thirdparty/spdlog)

set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/assimp)

add_subdirectory(src)
add_subdirectory(kernels)

add_executable(LuisaRender main.cpp)

target_link_libraries(LuisaRender PRIVATE luisarender::luisarender kernels)
