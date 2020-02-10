cmake_minimum_required(VERSION 3.15)
project(LuisaRender)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

include_directories(thirdparty)
include_directories(thirdparty/glm)

set(GLM_QUIET ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_CXX_17 ON CACHE BOOL "" FORCE)
add_compile_definitions(GLM_FORCE_AVX2)
add_compile_definitions(GLM_FORCE_CXX17)
add_subdirectory(thirdparty/glm)
link_libraries(glm_static)

set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
include_directories(thirdparty/assimp/include)
add_subdirectory(thirdparty/assimp)
link_libraries(assimp)

add_subdirectory(kernels)
include_directories(src)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h src/*.mm src/*.m)
add_executable(LuisaRender ${SOURCE_FILES} main.cpp)
add_dependencies(LuisaRender kernels)

target_link_libraries(LuisaRender "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")