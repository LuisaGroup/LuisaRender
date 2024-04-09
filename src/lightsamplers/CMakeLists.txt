add_library(luisa-render-lightsamplers INTERFACE)
luisa_render_add_plugin(uniform CATEGORY lightsampler SOURCES uniform.cpp)
luisa_render_add_plugin(power CATEGORY lightsampler SOURCES power.cpp)
