add_library(kernels INTERFACE)

file(GLOB_RECURSE SOURCE_FILES *.sk)
add_custom_target(kernel_sources DEPENDS ${SOURCE_FILES} SOURCES ${SOURCE_FILES})
add_dependencies(kernels kernel_sources)

add_library(luisarender::kernels ALIAS kernels)

install(FILES ${SOURCE_FILES} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/kernels")
