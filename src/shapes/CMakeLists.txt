luisa_render_add_plugin(instance CATALOG shapes SOURCES instance.cpp)
luisa_render_add_plugin(replica CATALOG shapes SOURCES replica.cpp)

luisa_render_add_plugin(triangle_mesh CATALOG shapes SOURCES triangle_mesh.cpp)

find_package(assimp REQUIRED)
target_link_libraries(triangle_mesh PRIVATE assimp::assimp)
