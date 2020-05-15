file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(util INTERFACE)
target_sources(util INTERFACE ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(util INTERFACE c_std_11 cxx_std_17)

find_package(spdlog REQUIRED)
target_link_libraries(util INTERFACE luisarender::headers spdlog::spdlog)
set_target_properties(util PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

add_library(luisarender::util ALIAS util)
install(TARGETS util PUBLIC_HEADER DESTINATION include/util)
