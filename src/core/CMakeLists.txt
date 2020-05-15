file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(core STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(core PUBLIC c_std_11 cxx_std_17)

find_package(glm REQUIRED)
target_link_libraries(core PUBLIC glm::glm luisarender::util luisarender::headers)
target_compile_definitions(core PUBLIC GLM_FORCE_AVX2 GLM_FORCE_CXX17)
set_target_properties(core PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

add_library(luisarender::core ALIAS core)
install(TARGETS core ARCHIVE DESTINATION lib/core PUBLIC_HEADER DESTINATION include/core)
