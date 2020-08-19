file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(core SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(core PUBLIC c_std_11 cxx_std_17)

target_link_libraries(core PUBLIC glm_config spdlog_config luisarender::headers)
set_target_properties(core PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(luisarender::core ALIAS core)
install(TARGETS core LIBRARY DESTINATION bin PUBLIC_HEADER DESTINATION include/core)
