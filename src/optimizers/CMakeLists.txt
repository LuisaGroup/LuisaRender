add_library(luisa-render-optimizers INTERFACE)
luisa_render_add_plugin(gd CATEGORY optimizer SOURCES gd.cpp)
luisa_render_add_plugin(adam CATEGORY optimizer SOURCES adam.cpp)
