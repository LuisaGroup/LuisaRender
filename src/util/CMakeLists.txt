file(GLOB SOURCE_FILES *.h *.cpp)

add_library(util SHARED ${SOURCE_FILES})
target_compile_features(util PUBLIC c_std_11 cxx_std_17)

find_package(spdlog REQUIRED)
target_link_libraries(util PUBLIC luisarender::headers spdlog::spdlog)

add_library(luisarender::util ALIAS util)
