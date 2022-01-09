add_library(luisa-render-integrators INTERFACE)
luisa_render_add_plugin(normal CATEGORY integrator SOURCES normal.cpp)
luisa_render_add_plugin(megapath CATEGORY integrator SOURCES megakernel_path_tracing.cpp)
