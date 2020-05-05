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

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(kernels)

add_executable(LuisaRender main.cpp)
target_link_libraries(LuisaRender PRIVATE luisarender::luisarender luisarender::kernels)
