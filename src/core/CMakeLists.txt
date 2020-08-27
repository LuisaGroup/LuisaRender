file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisacore SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(luisacore PUBLIC c_std_11 cxx_std_17)

target_link_libraries(luisacore PUBLIC glm_config spdlog_config luisaheaders)
set_target_properties(luisacore PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")

add_library(luisarender::core ALIAS luisacore)
install(TARGETS luisacore DESTINATION "bin/" LIBRARY DESTINATION "bin/" PUBLIC_HEADER DESTINATION "include/core/")
