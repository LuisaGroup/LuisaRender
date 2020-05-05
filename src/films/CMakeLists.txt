luisa_render_add_plugin(rgb SOURCES rgb.cpp rgb.h)

find_package(OpenCV REQUIRED)
target_link_libraries(rgb PRIVATE ${OpenCV_LIBS})
