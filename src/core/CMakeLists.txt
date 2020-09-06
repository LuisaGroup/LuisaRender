file(GLOB HEADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisa-core SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(luisa-core PUBLIC c_std_11 cxx_std_17)
target_compile_definitions(luisa-core PUBLIC NOMINMAX)

target_link_libraries(luisa-core PUBLIC spdlog_config cxxopts_config luisa-headers)
set_target_properties(luisa-core PROPERTIES
                      PUBLIC_HEADER "${HEADER_FILES}"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(luisa::core ALIAS luisa-core)

foreach (HEADER_FILE ${HEADER_FILES})
    configure_file(${HEADER_FILE} "${CMAKE_BINARY_DIR}/include/core/${HEADER_FILE}" COPYONLY)
endforeach ()
