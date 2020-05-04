file(GLOB SOURCE_FILES *.cpp *.h)

add_library(core SHARED ${SOURCE_FILES})
target_compile_features(core PUBLIC c_std_11 cxx_std_17)
target_link_libraries(core PUBLIC ${OpenCV_LIBS} glm::glm assimp::assimp luisarender::util luisarender::headers)

add_library(luisarender::core ALIAS core)
