add_library(luisaheaders INTERFACE)
target_include_directories(luisaheaders INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(luisaheaders INTERFACE c_std_11 cxx_std_17)
add_library(luisa::headers ALIAS luisaheaders)

add_subdirectory(ext/spdlog)
add_library(spdlog_config INTERFACE)
target_link_libraries(spdlog_config INTERFACE spdlog::spdlog)

add_library(jitify_config INTERFACE)
target_include_directories(jitify_config INTERFACE ext)

add_subdirectory(core)
add_subdirectory(compute)
add_subdirectory(backends)

add_library(luisarender INTERFACE)
target_link_libraries(luisarender INTERFACE luisacore luisacompute luisabackends)
add_library(luisarender::luisarender ALIAS luisarender)

function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisarender)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
endfunction()

add_subdirectory(tests)
add_subdirectory(apps)
