add_library(luisa-headers INTERFACE)
target_include_directories(luisa-headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(luisa-headers INTERFACE c_std_11 cxx_std_20)
add_library(luisa::headers ALIAS luisa-headers)

add_subdirectory(ext/spdlog)
add_library(spdlog_config INTERFACE)
target_link_libraries(spdlog_config INTERFACE spdlog::spdlog)

add_library(jitify_config INTERFACE)
target_include_directories(jitify_config INTERFACE ext)

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/cxxopts)
add_library(cxxopts_config INTERFACE)
target_link_libraries(cxxopts_config INTERFACE cxxopts::cxxopts)

add_library(tinyobjloader_config INTERFACE)
target_include_directories(tinyobjloader_config INTERFACE ext)

add_library(tinyexr_config INTERFACE)
target_include_directories(tinyexr_config INTERFACE ext)

add_library(stb_config INTERFACE)
target_include_directories(stb_config INTERFACE ext)

add_subdirectory(core)
add_subdirectory(compute)
add_subdirectory(backends)
add_subdirectory(render)

add_library(luisa-plugins INTERFACE)
function(luisa_render_add_plugin name)
    cmake_parse_arguments(PLUGIN "" "CATEGORY" "SOURCES" ${ARGN})
    add_library(${name} MODULE ${PLUGIN_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisarender-render)
    add_dependencies(luisa-plugins ${name})
    list(FILTER PLUGIN_SOURCES INCLUDE REGEX ".*\\.h")
    set_target_properties(${name} PROPERTIES PUBLIC_HEADER "${PLUGIN_SOURCES}")
endfunction()

add_library(luisa-all INTERFACE)
target_link_libraries(luisa-all INTERFACE luisa-core luisa-compute luisa-backends luisa-render)
add_library(luisa::all ALIAS luisa-all)

function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisa-all)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
endfunction()

add_subdirectory(tests)
add_subdirectory(apps)
