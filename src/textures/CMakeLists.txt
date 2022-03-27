add_library(luisa-render-textures INTERFACE)
luisa_render_add_plugin(constant CATEGORY texture SOURCES constant.cpp)
luisa_render_add_plugin(image CATEGORY texture SOURCES image.cpp)
