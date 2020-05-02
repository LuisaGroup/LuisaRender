cmake_minimum_required(VERSION 3.16...3.17)
cmake_policy(VERSION 3.16)

project(LuisaRender LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    set(ENABLE_METAL_DEFAULT ON)
else ()
    set(ENABLE_METAL_DEFAULT OFF)
endif ()
option(ENABLE_METAL "Build with Apple Metal support" ${ENABLE_METAL_DEFAULT})

add_executable(LuisaRender)
target_compile_features(LuisaRender PRIVATE c_std_11 cxx_std_17)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h)
target_sources(LuisaRender PRIVATE ${SOURCE_FILES} main.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(LuisaRender PRIVATE ${OpenCV_LIBS})

set(GLM_QUIET ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_CXX_17 ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/glm)
target_compile_definitions(glm INTERFACE GLM_FORCE_AVX2 GLM_FORCE_CXX17)
target_link_libraries(LuisaRender PRIVATE glm::glm)

add_subdirectory(thirdparty/spdlog)
target_link_libraries(LuisaRender PRIVATE spdlog)

set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/assimp)
target_link_libraries(LuisaRender PRIVATE assimp)

target_include_directories(LuisaRender PRIVATE src)

add_subdirectory(kernels)
target_link_libraries(LuisaRender PRIVATE kernels)

if (ENABLE_METAL)
    enable_language(OBJC)
    enable_language(OBJCXX)
    file(GLOB OC_SOURCE_FILES src/backends/metal/*.mm src/backends/metal/*.h)
    target_sources(LuisaRender PRIVATE ${OC_SOURCE_FILES})
    target_compile_options(LuisaRender PRIVATE $<$<OR:$<COMPILE_LANGUAGE:OBJC>,$<COMPILE_LANGUAGE:OBJCXX>>:-fobjc-arc>)
    target_link_libraries(LuisaRender PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
endif ()

install(DIRECTORY src/core/ DESTINATION include/core FILES_MATCHING PATTERN "*.h")
