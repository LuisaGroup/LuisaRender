file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisa-compute SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(luisa-compute PUBLIC luisa-core tinyexr_config stb_config)

set_target_properties(luisa-compute PROPERTIES
                      PUBLIC_HEADER "${HEADER_FILES}"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(luisa::compute ALIAS luisa-compute)
