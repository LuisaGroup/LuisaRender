add_library(luisa-render-films INTERFACE)

luisa_render_add_plugin(color CATEGORY film SOURCES color.cpp)

find_package(OpenCV CONFIG)
if (OpenCV_FOUND)
    target_link_libraries(luisa-render-film-color PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(luisa-render-film-color PUBLIC LUISA_USE_OPENCV)
endif ()
