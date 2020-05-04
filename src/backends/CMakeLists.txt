add_library(backends INTERFACE)

add_subdirectory(metal)
target_link_libraries(backends INTERFACE metal_backend)

add_library(luisarender::backends ALIAS backends)
