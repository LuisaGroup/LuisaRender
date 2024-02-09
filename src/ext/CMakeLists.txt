add_library(luisa-render-ext INTERFACE)

add_subdirectory(cxxopts)
target_link_libraries(luisa-render-ext INTERFACE cxxopts::cxxopts)

add_library(fast_float INTERFACE)
target_include_directories(fast_float INTERFACE fast_float/include)
target_link_libraries(luisa-render-ext INTERFACE fast_float)

# shared by assimp and tinyexr
find_package(ZLIB)

# Assimp
if (ZLIB_FOUND)
    # work around assimp's broken install when using system zlib
    if (NOT TARGET zlib)
        add_library(zlib INTERFACE)
    endif ()
    if (NOT TARGET zlibstatic)
        add_library(zlibstatic INTERFACE)
    endif ()
    set(ASSIMP_BUILD_ZLIB OFF CACHE BOOL "" FORCE)
else ()
    set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
endif ()
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL ON CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT ON CACHE BOOL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0)
    add_compile_options(-Wno-deprecated-non-prototype)
endif ()
add_subdirectory(assimp)
set_target_properties(assimp PROPERTIES
        UNITY_BUILD OFF
        OUTPUT_NAME "luisa-render-ext-assimp")
if (UNIX AND NOT APPLE) # TODO: fix this
    target_compile_definitions(assimp PRIVATE USE_FILE32API=1)
endif ()
if (ASSIMP_BUILD_ZLIB)
    if (TARGET zlib)
        set_target_properties(zlib PROPERTIES
                WINDOWS_EXPORT_ALL_SYMBOLS ON
                OUTPUT_NAME "luisa-render-ext-zlib")
        install(TARGETS zlib
                LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif ()
endif ()
target_link_libraries(luisa-render-ext INTERFACE assimp::assimp)

# tinyexr
set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
if (ZLIB_FOUND)
    add_library(tinyexr SHARED tinyexr.cpp)
    target_compile_definitions(tinyexr PUBLIC TINYEXR_USE_MINIZ=0)
    target_link_libraries(tinyexr PRIVATE ZLIB::ZLIB)
else ()
    add_library(tinyexr SHARED tinyexr.cpp tinyexr/deps/miniz/miniz.c)
    target_compile_definitions(tinyexr PUBLIC TINYEXR_USE_MINIZ=1)
    target_include_directories(tinyexr PRIVATE tinyexr/deps/miniz)
endif ()
target_include_directories(tinyexr PUBLIC tinyexr)
target_link_libraries(tinyexr PUBLIC ${CMAKE_DL_LIBS})
set_target_properties(tinyexr PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        OUTPUT_NAME "luisa-render-ext-tinyexr")
target_link_libraries(luisa-render-ext INTERFACE tinyexr)
install(TARGETS tinyexr
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# nlohmann json
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE json/single_include)
target_link_libraries(luisa-render-ext INTERFACE nlohmann_json)
