add_library(luisa-render-textures INTERFACE)
luisa_render_add_plugin(constant CATEGORY texture SOURCES constant.cpp)
luisa_render_add_plugin(image CATEGORY texture SOURCES image.cpp)
luisa_render_add_plugin(swizzle CATEGORY texture SOURCES swizzle.cpp)
luisa_render_add_plugin(checkerboard CATEGORY texture SOURCES checkerboard.cpp)

# sky texture precomputation
add_library(luisa-render-texture-sky-precompute SHARED nishita_precompute.cpp sky_precompute.h)
target_link_libraries(luisa-render-texture-sky-precompute PUBLIC luisa-render-util luisa-render-base)
set_target_properties(luisa-render-texture-sky-precompute PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Nishita sky texture
luisa_render_add_plugin(nishitasky CATEGORY texture SOURCES nishita_sky.cpp)
target_link_libraries(luisa-render-texture-nishitasky PRIVATE luisa-render-texture-sky-precompute)
