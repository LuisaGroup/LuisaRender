luisa_render_add_plugin(rgb CATEGORY films SOURCES rgb.cpp)

find_package(OpenEXR REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)
find_library(OpenImageIO_HEIF_LIBRARY heif REQUIRED)
target_link_libraries(rgb PRIVATE OpenImageIO::OpenImageIO ${OpenImageIO_HEIF_LIBRARY} ${OpenEXR_LIBRARIES})
