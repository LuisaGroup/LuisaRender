file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)

add_library(luisarender SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(luisarender PUBLIC luisacompute)

add_library(luisa::render ALIAS luisarender)
