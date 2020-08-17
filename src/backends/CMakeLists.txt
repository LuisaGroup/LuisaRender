add_library(backends INTERFACE)

function(luisa_render_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(${name} MODULE ${BACKEND_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisarender::core luisarender::compute)
    add_dependencies(backends ${name})
    list(FILTER BACKEND_SOURCES INCLUDE REGEX ".*\\.h")
    set_target_properties(${name} PROPERTIES PUBLIC_HEADER "${BACKEND_SOURCES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/backends")
    install(TARGETS ${name} LIBRARY DESTINATION lib/backends PUBLIC_HEADER DESTINATION include/backends/${name})
endfunction()

if (LUISA_RENDER_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

add_library(luisarender::backends ALIAS backends)
