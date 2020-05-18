luisa_render_add_plugin(instance CATEGORY shapes SOURCES instance.cpp)
luisa_render_add_plugin(replica CATEGORY shapes SOURCES replica.cpp)

luisa_render_add_plugin(triangle_mesh CATEGORY shapes SOURCES triangle_mesh.cpp)

find_package(assimp REQUIRED)
target_link_libraries(triangle_mesh PRIVATE assimp::assimp)
