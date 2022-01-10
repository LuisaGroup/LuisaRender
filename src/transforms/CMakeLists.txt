add_library(luisa-render-transforms INTERFACE)
luisa_render_add_plugin(identity CATEGORY transform SOURCES identity.cpp)
luisa_render_add_plugin(srt CATEGORY transform SOURCES srt.cpp)
