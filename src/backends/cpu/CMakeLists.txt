file(GLOB SOURCE_FILES *.h *.cpp)
luisa_render_add_backend(cpu SOURCES ${SOURCE_FILES})
