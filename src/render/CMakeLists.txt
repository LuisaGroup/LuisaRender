file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(render STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(render PUBLIC c_std_11 cxx_std_17)

target_link_libraries(render PUBLIC luisarender::headers luisarender::compute)
set_target_properties(render PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

add_library(luisarender::render ALIAS render)
install(TARGETS render ARCHIVE DESTINATION lib/render PUBLIC_HEADER DESTINATION include/render)
