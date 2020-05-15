luisa_render_add_plugin(rgb CATALOG films SOURCES rgb.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(rgb PRIVATE ${OpenCV_LIBS})
