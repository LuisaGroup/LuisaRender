add_library(luisa-render-environments INTERFACE)
luisa_render_add_plugin(black CATEGORY environment SOURCES black.cpp)
luisa_render_add_plugin(map CATEGORY environment SOURCES map.cpp)
