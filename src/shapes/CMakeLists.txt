add_library(luisa-render-shapes INTERFACE)
luisa_render_add_plugin(trianglemesh CATEGORY shape SOURCES triangle_mesh.cpp)
luisa_render_add_plugin(instance CATEGORY shape SOURCES instance.cpp)
luisa_render_add_plugin(group CATEGORY shape SOURCES group.cpp)
