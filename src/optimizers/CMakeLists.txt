add_library(luisa-render-optimizers INTERFACE)
luisa_render_add_plugin(gd CATEGORY optimizer SOURCES GD.cpp)
luisa_render_add_plugin(adam CATEGORY optimizer SOURCES Adam.cpp)