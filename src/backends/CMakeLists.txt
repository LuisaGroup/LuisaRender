add_library(luisa-backends INTERFACE)

function(luisa_render_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(luisa-backend-${name} MODULE ${BACKEND_SOURCES})
    target_compile_features(luisa-backend-${name} PRIVATE c_std_11 cxx_std_17)
    target_link_libraries(luisa-backend-${name} PRIVATE mimalloc_config luisa-compute)
    add_dependencies(luisa-backends luisa-backend-${name})
    set_target_properties(luisa-backend-${name} PROPERTIES
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/backends/"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/backends/")
endfunction()

if (LUISA_RENDER_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

if (LUISA_RENDER_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

if (LUISA_RENDER_ENABLE_OPENCL)
    add_subdirectory(opencl)
endif ()

add_library(luisa::backends ALIAS luisa-backends)
