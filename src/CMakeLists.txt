if (WIN32)
    add_compile_definitions(
            NOMINMAX=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE=1
            _CRT_SECURE_NO_WARNINGS=1
            _UNICODE=1)
    set(CMAKE_SHARED_MODULE_PREFIX "") # for MinGW/Cygwin etc.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/MP)
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0)
    add_compile_options(-Wno-deprecated-builtins)
endif ()

add_subdirectory(ext)

add_library(luisa-render-include INTERFACE)
target_include_directories(luisa-render-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-render-include INTERFACE c_std_11 cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-render-include INTERFACE
            /wd4068 # Maxwell loves custom pragmas
            /wd5105 # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor)
endif ()

set(LUISA_COMPUTE_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_UNITY_BUILD ${LUISA_RENDER_ENABLE_UNITY_BUILD} CACHE BOOL "" FORCE)

set(SKBUILD_OLD ${SKBUILD})
set(SKBUILD OFF)
add_subdirectory(compute)
set(SKBUILD ${SKBUILD_OLD})

target_include_directories(luisa-render-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/luisa)

add_subdirectory(sdl)
add_subdirectory(util)
add_subdirectory(base)

function(luisa_render_add_plugin name)
    cmake_parse_arguments(PLUGIN "" "CATEGORY" "SOURCES" ${ARGN})
    set(lib_name luisa-render-${PLUGIN_CATEGORY}-${name})
    add_library(${lib_name} MODULE ${PLUGIN_SOURCES})
    target_link_libraries(${lib_name} PRIVATE luisa-render-util luisa-render-base)
    target_compile_definitions(${lib_name} PRIVATE LUISA_RENDER_PLUGIN_NAME="${name}")
    if (${PLUGIN_CATEGORY} STREQUAL "spectrum")
        set(plugin_lib_name luisa-render-spectra)
    elseif(${PLUGIN_CATEGORY} STREQUAL "medium")
        set(plugin_lib_name luisa-render-media)
    else ()
        set(plugin_lib_name luisa-render-${PLUGIN_CATEGORY}s)
    endif ()
    add_dependencies(${plugin_lib_name} ${lib_name})
    set_target_properties(${lib_name} PROPERTIES
            UNITY_BUILD OFF
            DEBUG_POSTFIX "")
    install(TARGETS ${lib_name}
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

add_subdirectory(films)
add_subdirectory(media)
add_subdirectory(shapes)
add_subdirectory(lights)
add_subdirectory(spectra)
add_subdirectory(cameras)
add_subdirectory(filters)
add_subdirectory(textures)
add_subdirectory(samplers)
add_subdirectory(surfaces)
add_subdirectory(transforms)
add_subdirectory(integrators)
add_subdirectory(environments)
add_subdirectory(lightsamplers)
add_subdirectory(phasefunctions)
add_subdirectory(texturemappings)
add_library(luisa-render INTERFACE)

target_link_libraries(luisa-render INTERFACE
        luisa-render-ext
        luisa-render-sdl
        luisa-render-base
        luisa-render-util
        luisa-render-films
        luisa-render-media
        luisa-render-shapes
        luisa-render-lights
        luisa-render-filters
        luisa-render-spectra
        luisa-render-cameras
        luisa-render-textures
        luisa-render-samplers
        luisa-render-surfaces
        luisa-render-transforms
        luisa-render-integrators
        luisa-render-environments
        luisa-render-lightsamplers
        luisa-render-phasefunctions
        luisa-render-texturemappings)
add_library(luisa::render ALIAS luisa-render)

add_subdirectory(apps)

if (LUISA_RENDER_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (SKBUILD)
    add_subdirectory(python)
endif ()
