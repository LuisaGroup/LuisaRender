file(GLOB SOURCE_FILES *.cpp *.h)
luisa_render_add_application(RDFC SOURCES ${SOURCE_FILES})
target_link_libraries(RDFC PRIVATE ${OpenCV_LIBS})
