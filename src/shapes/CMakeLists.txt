luisa_render_add_plugin(group CATEGORY shape SOURCES group.cpp)
luisa_render_add_plugin(instance CATEGORY shape SOURCES instance.cpp)
luisa_render_add_plugin(mesh CATEGORY shape SOURCES mesh.cpp)
target_link_libraries(luisa-shape-mesh PRIVATE assimp_config)
