add_library(luisa-render-ext INTERFACE)

add_subdirectory(cxxopts)
target_link_libraries(luisa-render-ext INTERFACE cxxopts::cxxopts)

add_library(fast_float INTERFACE)
target_include_directories(fast_float INTERFACE fast_float/include)
target_link_libraries(luisa-render-ext INTERFACE fast_float)

#find_package(assimp CONFIG)
if (NOT assimp_FOUND)
    set(ASSIMP_BUILD_ZLIB ${WIN32} CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
    set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT ON CACHE BOOL "" FORCE)
    add_subdirectory(assimp)
    set_target_properties(assimp PROPERTIES UNITY_BUILD OFF)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0)
        target_compile_options(assimp PRIVATE -Wno-deprecated-non-prototype)
    endif ()
endif ()
target_link_libraries(luisa-render-ext INTERFACE assimp::assimp)

# tinyexr & miniz
add_library(tinyexr SHARED tinyexr/tinyexr.cc tinyexr/deps/miniz/miniz.c)
target_include_directories(tinyexr PUBLIC tinyexr tinyexr/deps/miniz)
target_link_libraries(tinyexr PUBLIC ${CMAKE_DL_LIBS})
set_target_properties(tinyexr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(luisa-render-ext INTERFACE tinyexr)
