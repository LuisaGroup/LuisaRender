luisa_render_add_plugin(rgb CATEGORY films SOURCES rgb.cpp)

find_package(OpenImageIO REQUIRED)
target_link_libraries(rgb PRIVATE OpenImageIO::OpenImageIO)
#find_package(OpenCV REQUIRED)
target_link_libraries(rgb PRIVATE ${OpenCV_LIBS})
