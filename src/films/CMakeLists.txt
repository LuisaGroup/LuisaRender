luisa_render_add_plugin(rgb_film SOURCES rgb_film.cpp rgb_film.h)

find_package(OpenCV REQUIRED)
target_link_libraries(rgb_film PRIVATE ${OpenCV_LIBS})
