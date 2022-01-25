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

add_subdirectory(compute)
add_subdirectory(sdl)
add_subdirectory(util)
add_subdirectory(base)

function(luisa_render_add_plugin name)
    cmake_parse_arguments(PLUGIN "" "CATEGORY" "SOURCES" ${ARGN})
    set(lib_name luisa-render-${PLUGIN_CATEGORY}-${name})
    add_library(${lib_name} MODULE ${PLUGIN_SOURCES})
    target_link_libraries(${lib_name} PRIVATE luisa-render-base)
    add_dependencies(luisa-render-${PLUGIN_CATEGORY}s ${lib_name})
    set_target_properties(${lib_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
            UNITY_BUILD ON
            DEBUG_POSTFIX "")
    foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set_target_properties(${lib_name} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/"
                RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/"
                UNITY_BUILD ON)
    endforeach ()
endfunction()

add_subdirectory(cameras)
add_subdirectory(films)
add_subdirectory(filters)
add_subdirectory(integrators)
add_subdirectory(materials)
add_subdirectory(samplers)
add_subdirectory(shapes)
add_subdirectory(lights)
add_subdirectory(transforms)
add_subdirectory(environments)
add_subdirectory(lightsamplers)

add_library(luisa-render INTERFACE)
target_link_libraries(luisa-render INTERFACE
        luisa-render-ext
        luisa-render-include
        luisa-render-sdl
        luisa-render-base
        luisa-render-util
        luisa-render-cameras
        luisa-render-films
        luisa-render-filters
        luisa-render-integrators
        luisa-render-materials
        luisa-render-samplers
        luisa-render-shapes
        luisa-render-lights
        luisa-render-transforms
        luisa-render-environments
        luisa-render-lightsamplers)
add_library(luisa::render ALIAS luisa-render)

add_subdirectory(apps)
