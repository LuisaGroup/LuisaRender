add_library(backends INTERFACE)

if (LUISA_RENDER_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

add_library(luisarender::backends ALIAS backends)
