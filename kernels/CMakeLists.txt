add_library(kernels INTERFACE)

if (LUISA_RENDER_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

add_library(luisarender::kernels ALIAS kernels)
