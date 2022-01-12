add_library(luisa-render-materials INTERFACE)
luisa_render_add_plugin(black CATEGORY material SOURCES black.cpp)
luisa_render_add_plugin(lambert CATEGORY material SOURCES lambert.cpp)
