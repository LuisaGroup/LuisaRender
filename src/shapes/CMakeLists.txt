luisa_render_add_plugin(group CATEGORY shape SOURCES group.cpp)

luisa_render_add_plugin(wavefront CATEGORY shape SOURCES wavefront.cpp)
target_link_libraries(luisa-shape-wavefront PRIVATE tinyobjloader_config)
