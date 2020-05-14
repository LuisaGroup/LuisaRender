luisa_render_add_plugin(instance SOURCES instance.cpp)
luisa_render_add_plugin(replica SOURCES replica.cpp)

luisa_render_add_plugin(triangle_mesh SOURCES triangle_mesh.cpp)

find_package(assimp REQUIRED)
target_link_libraries(triangle_mesh PRIVATE assimp::assimp)
