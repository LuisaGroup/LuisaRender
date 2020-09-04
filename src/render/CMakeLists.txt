file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisarender SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(luisarender PUBLIC luisacore luisacompute)
set_target_properties(luisarender PROPERTIES
                      PUBLIC_HEADER "${HEADER_FILES}"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)
                      
add_library(luisa::render ALIAS luisarender)
