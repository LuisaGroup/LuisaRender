add_library(luisa-render-spectra INTERFACE)
luisa_render_add_plugin(rgb CATEGORY spectrum SOURCES rgb.cpp)
luisa_render_add_plugin(hero CATEGORY spectrum SOURCES hero.cpp)
