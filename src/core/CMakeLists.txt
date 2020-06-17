file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(core STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(core PUBLIC c_std_11 cxx_std_17)

target_link_libraries(core PUBLIC luisarender::compute)
set_target_properties(core PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

add_library(luisarender::core ALIAS core)
install(TARGETS core ARCHIVE DESTINATION lib/core PUBLIC_HEADER DESTINATION include/core)
