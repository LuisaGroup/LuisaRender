file(GLOB METAL_KERNEL_SOURCE_FILES metal/*.h metal/*.metal)

file(GLOB KERNEL_SOURCE_FILES ${METAL_KERNEL_SOURCE_FILES})
add_custom_target(kernels ALL SOURCES ${KERNEL_SOURCE_FILES})

add_custom_command(TARGET kernels COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/build_metal_kernels.py)
