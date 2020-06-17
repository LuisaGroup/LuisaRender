file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(compute STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(compute PUBLIC c_std_11 cxx_std_17)

find_package(glm REQUIRED)
target_link_libraries(compute PUBLIC glm::glm luisarender::util luisarender::headers)
target_compile_definitions(compute PUBLIC GLM_FORCE_AVX2 GLM_FORCE_CXX17)
set_target_properties(compute PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

add_library(luisarender::compute ALIAS compute)
install(TARGETS compute ARCHIVE DESTINATION lib/compute PUBLIC_HEADER DESTINATION include/compute)
