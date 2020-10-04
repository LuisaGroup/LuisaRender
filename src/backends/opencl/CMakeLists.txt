message(STATUS "Building with OpenCL support")

file(GLOB SOURCE_FILES *.h *.cpp)
luisa_render_add_backend(opencl SOURCES ${SOURCE_FILES})
target_link_libraries(luisa-backend-opencl PRIVATE opencl_config)
