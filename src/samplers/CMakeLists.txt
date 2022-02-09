add_library(luisa-render-samplers INTERFACE)
luisa_render_add_plugin(independent CATEGORY sampler SOURCES independent.cpp)
luisa_render_add_plugin(sobol CATEGORY sampler SOURCES sobol.cpp)
