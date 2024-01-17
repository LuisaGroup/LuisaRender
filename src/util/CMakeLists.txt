add_library(luisa-render-util SHARED
        sampling.cpp sampling.h
        frame.cpp frame.h
        imageio.cpp imageio.h
        xform.cpp xform.h
        spec.cpp spec.h
        colorspace.h
        half.cpp half.h
        u64.cpp u64.h
        rng.cpp rng.h
        ies.cpp ies.h
        scattering.cpp scattering.h
        bluenoise.cpp bluenoise.h
        sobolmatrices.cpp sobolmatrices.h
        pmj02tables.cpp pmj02tables.h
        complex.h
        medium_tracker.cpp medium_tracker.h
        progress_bar.cpp progress_bar.h
        loop_subdiv.cpp loop_subdiv.h
        vertex.h
        counter_buffer.cpp counter_buffer.h
        polymorphic_closure.h
        command_buffer.cpp command_buffer.h
        thread_pool.cpp thread_pool.h)
target_link_libraries(luisa-render-util PUBLIC
        luisa::compute
        luisa-render-include
        luisa-render-ext)
set_target_properties(luisa-render-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_RENDER_ENABLE_UNITY_BUILD})
install(TARGETS luisa-render-util
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})