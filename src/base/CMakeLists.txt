set(LUISA_RENDER_BASE_SOURCES
        scene.cpp scene.h
        transform.h
        camera.cpp camera.h
        sampler.cpp sampler.h)

add_library(luisa-render-base SHARED ${LUISA_RENDER_BASE_SOURCES})
target_link_libraries(luisa-render-base PUBLIC
        luisa-render-include
        luisa-render-ext
        luisa::compute)
set_target_properties(luisa-render-base PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
