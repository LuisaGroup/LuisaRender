add_library(luisa-render-lights INTERFACE)
luisa_render_add_plugin(black CATEGORY light SOURCES black.cpp)
luisa_render_add_plugin(diffuse CATEGORY light SOURCES diffuse.cpp)
luisa_render_add_plugin(fakepoint CATEGORY light SOURCES fake_point.cpp)
