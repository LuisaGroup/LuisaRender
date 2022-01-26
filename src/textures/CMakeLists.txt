add_library(luisa-render-textures INTERFACE)
luisa_render_add_plugin(constant CATEGORY texture SOURCES constant.cpp)
luisa_render_add_plugin(constantilluminant CATEGORY texture SOURCES constant_illuminant.cpp)
