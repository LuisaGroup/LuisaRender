add_library(luisa-render-films INTERFACE)

find_package(OpenCV CONFIG)
if (OpenCV_FOUND)
    luisa_render_add_plugin(color CATEGORY film SOURCES color.cpp)
    target_link_libraries(luisa-render-film-color PRIVATE ${OpenCV_LIBS})
endif ()
