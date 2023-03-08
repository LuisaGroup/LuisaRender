add_library(luisa-render-media INTERFACE)
luisa_render_add_plugin(null CATEGORY medium SOURCES null.cpp)
luisa_render_add_plugin(vacuum CATEGORY medium SOURCES vacuum.cpp)
luisa_render_add_plugin(homogeneous CATEGORY medium SOURCES homogeneous.cpp)
