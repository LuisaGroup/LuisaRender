add_library(luisabackends INTERFACE)

function(luisa_render_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(${name} MODULE ${BACKEND_SOURCES})
    target_compile_features(${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(${name} PRIVATE luisacore luisacompute)
    add_dependencies(luisabackends ${name})
    list(FILTER BACKEND_SOURCES INCLUDE REGEX ".*\\.h")
    set_target_properties(${name} PROPERTIES
                          PUBLIC_HEADER "${BACKEND_SOURCES}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/backends/"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/backends/"
                          WINDOWS_EXPORT_ALL_SYMBOLS ON)
endfunction()

if (LUISA_RENDER_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

if (LUISA_RENDER_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

add_library(luisarender::backends ALIAS luisabackends)
