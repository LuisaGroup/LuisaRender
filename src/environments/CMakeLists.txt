add_library(luisa-render-environments INTERFACE)
luisa_render_add_plugin(null CATEGORY environment SOURCES null.cpp)
luisa_render_add_plugin(map CATEGORY environment SOURCES map.cpp)
