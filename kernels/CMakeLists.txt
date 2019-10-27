file(GLOB KERNEL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.metal)

add_custom_target(kernels ALL SOURCES ${KERNEL_SOURCE_FILES})
add_custom_command(TARGET kernels COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/build_kernels.py)
