file(GLOB HEADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisa-core SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(luisa-core PUBLIC c_std_11 cxx_std_17)
target_compile_definitions(luisa-core PUBLIC NOMINMAX)

target_link_libraries(luisa-core PUBLIC spdlog_config cxxopts_config luisa-headers ${CMAKE_DL_LIBS})
set_target_properties(luisa-core PROPERTIES
                      PUBLIC_HEADER "${HEADER_FILES}"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_custom_command(TARGET luisa-core POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory
                   "${CMAKE_BINARY_DIR}/$<CONFIG>/include/core/")
add_custom_command(TARGET luisa-core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                   ${HEADER_FILES} "${CMAKE_BINARY_DIR}/$<CONFIG>/include/core/"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(luisa::core ALIAS luisa-core)
