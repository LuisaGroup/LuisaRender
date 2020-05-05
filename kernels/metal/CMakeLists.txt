file(GLOB KERNEL_SOURCE_FILES *.metal *.h)
add_custom_target(metal_kernels SOURCES ${KERNEL_SOURCE_FILES})
add_dependencies(kernels metal_kernels)
