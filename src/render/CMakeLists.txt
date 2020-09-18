file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)
file(GLOB SHADER_HEADER_FILES shaders/*.h)
file(GLOB SHADER_SOURCE_FILES shaders/*.cpp)

add_library(luisa-render SHARED ${HEADER_FILES} ${SOURCE_FILES} ${SHADER_HEADER_FILES} ${SHADER_SOURCE_FILES})
target_link_libraries(luisa-render PUBLIC luisa-core luisa-compute)
set_target_properties(luisa-render PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)
                      
add_library(luisa::render ALIAS luisa-render)
