add_library(luisa-render-losses INTERFACE)
luisa_render_add_plugin(l1 CATEGORY loss SOURCES l1.cpp)
luisa_render_add_plugin(l2 CATEGORY loss SOURCES l2.cpp)
