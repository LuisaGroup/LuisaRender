add_library(luisa-render-spectra INTERFACE)
luisa_render_add_plugin(srgb CATEGORY spectrum SOURCES srgb.cpp)
luisa_render_add_plugin(hero CATEGORY spectrum SOURCES hero.cpp srgb2spec.cpp)
