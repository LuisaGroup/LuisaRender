luisa_render_add_plugin(instance CATEGORY shapes SOURCES instance.cpp)
luisa_render_add_plugin(replica CATEGORY shapes SOURCES replica.cpp)

luisa_render_add_plugin(triangle_mesh CATEGORY shapes SOURCES triangle_mesh.cpp)

find_package(assimp CONFIG REQUIRED)
find_library(ASSIMP_ZLIB_LIBRARY z REQUIRED)
find_library(ASSIMP_IRRXML_LIBRARY IrrXML REQUIRED)
target_link_libraries(triangle_mesh PRIVATE assimp::assimp ${ASSIMP_ZLIB_LIBRARY} ${ASSIMP_IRRXML_LIBRARY})
