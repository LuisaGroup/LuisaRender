luisa_render_add_plugin(rgb CATEGORY films SOURCES rgb.cpp)

find_package(OpenEXR CONFIG REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)
target_link_libraries(rgb PRIVATE OpenImageIO::OpenImageIO)
