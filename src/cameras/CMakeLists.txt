add_library(luisa-render-cameras INTERFACE)

luisa_render_add_plugin(pinhole CATEGORY camera SOURCES pinhole.cpp)
luisa_render_add_plugin(thinlens CATEGORY camera SOURCES thinlens.cpp)
