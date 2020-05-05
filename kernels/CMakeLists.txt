add_library(kernels INTERFACE)

if (LUISA_RENDER_ENABLE_METAL)
    
    set(INCLUDE_DIRECTORY "${CMAKE_SOURCE_DIR}/src")
    
    # Build kernels
    file(GLOB KERNEL_SOURCE_FILES metal/*.metal)
    file(GLOB_RECURSE KERNEL_HEADER_FILES ${INCLUDE_DIRECTORY}/*.h)
    foreach (SOURCE_FILE ${KERNEL_SOURCE_FILES})
        get_filename_component(KERNEL_NAME ${SOURCE_FILE} NAME_WE)
        set(IR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/metal/build/${KERNEL_NAME}.air)
        set(LIBRARY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/metal/bin/${KERNEL_NAME}.metallib)
        list(APPEND IR_FILES ${IR_FILE})
        list(APPEND LIBRARY_FILES ${LIBRARY_FILE})
        add_custom_command(
                OUTPUT ${IR_FILE}
                DEPENDS ${SOURCE_FILE} ${KERNEL_HEADER_FILES}
                COMMAND xcrun ARGS -sdk macosx metal -Wall -Wextra -Wno-c++17-extensions -O3 -ffast-math -I ${INCLUDE_DIRECTORY} -c ${SOURCE_FILE} -o ${IR_FILE} -D DEVICE_COMPATIBLE
                COMMENT "Compiling Metal kernel: '${SOURCE_FILE}'")
        add_custom_command(
                OUTPUT ${LIBRARY_FILE}
                DEPENDS ${IR_FILE}
                COMMAND xcrun ARGS -sdk macosx metallib ${IR_FILE} -o ${LIBRARY_FILE}
                COMMENT "Archiving Metal IR file into library: '${IR_FILE}'")
    endforeach ()
    
    add_custom_target(metal_kernels DEPENDS ${LIBRARY_FILES} SOURCES ${KERNEL_SOURCE_FILES} metal/compatibility.h)
    add_dependencies(kernels metal_kernels)

    install(DIRECTORY metal/ DESTINATION kernels/metal FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY metal/ DESTINATION kernels/metal FILES_MATCHING PATTERN "*.metal")
    
endif ()

add_library(luisarender::kernels ALIAS kernels)
