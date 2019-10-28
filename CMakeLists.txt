cmake_minimum_required(VERSION 3.15)
project(Metalight)

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
add_subdirectory(thirdparty/glm)

add_subdirectory(thirdparty/tinyobjloader)

add_subdirectory(kernels)
include_directories(src)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h src/*.mm src/*.m)
add_executable(Metalight ${SOURCE_FILES} main.cpp)
add_dependencies(Metalight kernels)

target_link_libraries(Metalight glm_static tinyobjloader "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")