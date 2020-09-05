add_library(luisaheaders INTERFACE)
target_include_directories(luisaheaders INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(luisaheaders INTERFACE c_std_11 cxx_std_17)
add_library(luisa::headers ALIAS luisaheaders)

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

add_subdirectory(core)
add_subdirectory(compute)
add_subdirectory(backends)
add_subdirectory(render)

add_library(luisaall INTERFACE)
target_link_libraries(luisaall INTERFACE luisacore luisacompute luisabackends luisarender)
add_library(luisa::all ALIAS luisaall)

function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisaall)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
endfunction()

add_subdirectory(tests)
add_subdirectory(apps)
