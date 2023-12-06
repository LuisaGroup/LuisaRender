set(LUISA_RENDER_BASE_SOURCES
        scene.cpp scene.h
        transform.cpp transform.h
        camera.cpp camera.h
        sampler.cpp sampler.h
        filter.cpp filter.h
        scene_node.cpp scene_node.h
        shape.cpp shape.h
        surface.cpp surface.h
        film.cpp film.h
        integrator.cpp integrator.h
        environment.cpp environment.h
        light.cpp light.h
        pipeline.cpp pipeline.h
        interaction.cpp interaction.h
        light_sampler.cpp light_sampler.h
        texture.cpp texture.h
        texture_mapping.cpp texture_mapping.h
        differentiation.cpp differentiation.h
        spectrum.cpp spectrum.h
        loss.h loss.cpp
        optimizer.cpp optimizer.h
        geometry.cpp geometry.h
        spd.cpp spd.h
        medium.cpp medium.h
        phase_function.cpp phase_function.h)

add_library(luisa-render-base SHARED ${LUISA_RENDER_BASE_SOURCES})
target_link_libraries(luisa-render-base PUBLIC
        luisa::compute
        luisa-render-sdl
        luisa-render-util)
set_target_properties(luisa-render-base PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_RENDER_ENABLE_UNITY_BUILD})
