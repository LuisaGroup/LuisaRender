add_library(luisabackends INTERFACE)

function(luisa_render_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(luisabackend${name} MODULE ${BACKEND_SOURCES})
    target_compile_features(luisabackend${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(luisabackend${name} PRIVATE luisacore luisacompute)
    add_dependencies(luisabackends luisabackend${name})
    list(FILTER BACKEND_SOURCES INCLUDE REGEX ".*\\.h")
    set_target_properties(luisabackend${name} PROPERTIES
                          PUBLIC_HEADER "${BACKEND_SOURCES}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/backends/"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/backends/")
endfunction()

if (LUISA_RENDER_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

if (LUISA_RENDER_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

add_subdirectory(cpu)

add_library(luisarender::backends ALIAS luisabackends)
