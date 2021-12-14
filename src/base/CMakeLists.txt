set(LUISA_RENDER_BASE_SOURCES
        scene.cpp scene.h
        transform.h
        camera.cpp camera.h
        sampler.cpp sampler.h
        filter.cpp filter.h
        scene_node.cpp scene_node.h
        scene_desc.cpp scene_desc.h
        scene_desc_node.cpp scene_desc_node.h
        shape.cpp shape.h
        material.cpp material.h
        film.cpp film.h
        integrator.cpp integrator.h
        environment.cpp environment.h)

add_library(luisa-render-base SHARED ${LUISA_RENDER_BASE_SOURCES})
target_link_libraries(luisa-render-base PUBLIC
        luisa-render-include
        luisa-render-ext
        luisa::compute)
set_target_properties(luisa-render-base PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
