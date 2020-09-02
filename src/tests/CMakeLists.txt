luisa_render_add_application(test_buffer_io SOURCES test_buffer_io.cpp)

find_package(OpenCV CONFIG REQUIRED)

luisa_render_add_application(test_texture_io SOURCES test_texture_io.cpp)
target_link_libraries(test_texture_io PRIVATE ${OpenCV_LIBS})

luisa_render_add_application(test_blur SOURCES test_blur.cpp)
target_link_libraries(test_blur PRIVATE ${OpenCV_LIBS})

luisa_render_add_application(test_bitonic_sort SOURCES test_bitonic_sort.cpp)
