file(GLOB SOURCE_FILES *.cpp *.h *.sk)

add_library(core SHARED ${SOURCE_FILES})
target_compile_features(core PUBLIC c_std_11 cxx_std_17)

find_package(glm REQUIRED)
target_link_libraries(core PUBLIC glm luisarender::util luisarender::headers)
target_compile_definitions(core PUBLIC GLM_FORCE_AVX2 GLM_FORCE_CXX17)

add_library(luisarender::core ALIAS core)
