add_library(headers INTERFACE)
target_include_directories(headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(luisarender::headers ALIAS headers)

find_package(spdlog CONFIG REQUIRED)
add_library(spdlog_config INTERFACE)
target_link_libraries(spdlog_config INTERFACE spdlog::spdlog spdlog::spdlog_header_only)

find_package(glm CONFIG REQUIRED)
add_library(glm_config INTERFACE)
target_link_libraries(glm_config INTERFACE glm)
target_compile_definitions(glm_config INTERFACE GLM_FORCE_AVX2 GLM_FORCE_CXX17)

add_subdirectory(core)
add_subdirectory(compute)

add_subdirectory(backends)

add_library(luisarender INTERFACE)
target_link_libraries(luisarender INTERFACE luisarender::core luisarender::compute luisarender::backends)
add_library(luisarender::luisarender ALIAS luisarender)

function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisarender::luisarender)
    install(TARGETS ${name} DESTINATION bin)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

add_subdirectory(tests)
add_subdirectory(apps)
