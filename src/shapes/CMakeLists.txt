luisa_render_add_plugin(group CATEGORY shape SOURCES group.cpp)
luisa_render_add_plugin(instance CATEGORY shape SOURCES instance.cpp)

luisa_render_add_plugin(wavefront_obj CATEGORY shape SOURCES wavefront_obj.cpp)
target_link_libraries(luisa-shape-wavefront_obj PRIVATE tinyobjloader_config)
