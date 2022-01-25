set(LUISA_RENDER_SCENE_SOURCES
        scene.cpp scene.h
        transform.cpp transform.h
        camera.cpp camera.h
        sampler.cpp sampler.h
        filter.cpp filter.h
        scene_node.cpp scene_node.h
        shape.cpp shape.h
        material.cpp material.h
        film.cpp film.h
        integrator.cpp integrator.h
        environment.cpp environment.h
        light.cpp light.h
        pipeline.cpp pipeline.h
        interaction.h
        light_sampler.cpp light_sampler.h)

add_library(luisa-render-scene SHARED ${LUISA_RENDER_SCENE_SOURCES})
target_link_libraries(luisa-render-scene PUBLIC
        luisa::compute
        luisa-render-sdl
        luisa-render-util)
set_target_properties(luisa-render-scene PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
