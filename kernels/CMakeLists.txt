add_library(kernels INTERFACE)

if (LUISA_RENDER_ENABLE_METAL)
    
    set(INCLUDE_DIRECTORY "${CMAKE_SOURCE_DIR}/src")
    
    # Build kernels
    file(GLOB KERNEL_SOURCE_FILES metal/*.metal)
    file(GLOB_RECURSE KERNEL_HEADER_FILES ${INCLUDE_DIRECTORY}/*.h)
    foreach (SOURCE_FILE ${KERNEL_SOURCE_FILES})
        get_filename_component(KERNEL_NAME ${SOURCE_FILE} NAME_WE)
        set(IR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/metal/build/${KERNEL_NAME}.air)
        list(APPEND IR_FILES ${IR_FILE})
        add_custom_command(
                OUTPUT ${IR_FILE}
                DEPENDS ${SOURCE_FILE} ${KERNEL_HEADER_FILES}
                COMMAND xcrun ARGS -sdk macosx metal -Wall -Wextra -Wno-c++17-extensions -O3 -ffast-math -I ${INCLUDE_DIRECTORY} -c ${SOURCE_FILE} -o ${IR_FILE} -D DEVICE_COMPATIBLE
                COMMENT "Compiling Metal kernel: '${SOURCE_FILE}'")
    endforeach ()
    
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/metal/bin/kernels.metallib
            DEPENDS ${IR_FILES}
            COMMAND xcrun ARGS -sdk macosx metallib ${IR_FILES} -o ${CMAKE_CURRENT_SOURCE_DIR}/metal/bin/kernels.metallib
            COMMENT "Archiving Metal IR files into library")
    
    add_custom_target(metal_kernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/bin/kernels.metallib SOURCES ${KERNEL_SOURCE_FILES} metal/compatibility.h)
    add_dependencies(kernels metal_kernels)

    install(DIRECTORY metal/ DESTINATION kernels/metal FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY metal/ DESTINATION kernels/metal FILES_MATCHING PATTERN "*.metal")
    
endif ()
