add_library(luisa-render-lights INTERFACE)
luisa_render_add_plugin(null CATEGORY light SOURCES null.cpp)
luisa_render_add_plugin(diffuse CATEGORY light SOURCES diffuse.cpp)
