set(LUISA_RENDER_AUTODIFF_SOURCES
        node_ad.cpp node_ad.h)

add_library(luisa-render-autodiff SHARED ${LUISA_RENDER_AUTODIFF_SOURCES})
target_link_libraries(luisa-render-autodiff PUBLIC
        luisa::compute
        luisa-render-include)
set_target_properties(luisa-render-autodiff PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
