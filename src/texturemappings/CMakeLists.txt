add_library(luisa-render-texturemappings INTERFACE)
luisa_render_add_plugin(uv CATEGORY texturemapping SOURCES uv.cpp)
luisa_render_add_plugin(spherical CATEGORY texturemapping SOURCES spherical.cpp)
