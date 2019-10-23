file(GLOB KERNEL_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.metal)
file(GLOB KERNEL_IR_FILES_OLD RELATIVE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/*.air)

add_custom_target(kernels ALL SOURCES ${KERNEL_SOURCE_FILES})

foreach (IR_FILE ${KERNEL_IR_FILES_OLD})
    add_custom_command(TARGET kernels PRE_BUILD
                       COMMAND rm ${CMAKE_CURRENT_BINARY_DIR}/${IR_FILE}
                       COMMENT "Cleaning old IR file: ${CMAKE_CURRENT_BINARY_DIR}/${IR_FILE}")
endforeach ()

foreach (SOURCE_FILE ${KERNEL_SOURCE_FILES})
    add_custom_command(TARGET kernels
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                       COMMAND xcrun -sdk macosx metal -std=macos-metal2.2 -Ofast -Wall -Wextra -ffast-math -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE}.air
                       COMMENT "Generating Metal IR file: ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE}.air")
endforeach ()

add_custom_command(TARGET kernels
                   COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/*.air -o ${CMAKE_BINARY_DIR}/bin/kernels.metallib
                   COMMENT "Generating Metal Library: ${CMAKE_BINARY_DIR}/bin/kernels.metallib")
