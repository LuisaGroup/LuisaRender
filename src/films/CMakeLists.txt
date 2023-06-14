add_library(luisa-render-films INTERFACE)
luisa_render_add_plugin(color CATEGORY film SOURCES color.cpp)
luisa_render_add_plugin(display CATEGORY film SOURCES display.cpp)
