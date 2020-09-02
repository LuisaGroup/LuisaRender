file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisacore SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(luisacore PUBLIC c_std_11 cxx_std_17)
target_compile_definitions(luisacore PUBLIC NOMINMAX)

target_link_libraries(luisacore PUBLIC spdlog_config cxxopts_config luisaheaders)
set_target_properties(luisacore PROPERTIES
                      PUBLIC_HEADER "${HEADER_FILES}"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(luisarender::core ALIAS luisacore)
file(COPY "${CMAKE_CURRENT_LIST_DIR}" DESTINATION "${CMAKE_BINARY_DIR}/include/")
