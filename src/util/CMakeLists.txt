add_library(luisa-render-util SHARED
        sampling.cpp sampling.h
        frame.cpp frame.h
        imageio.cpp imageio.h
        xform.cpp xform.h
        spec.cpp spec.h
        colorspace.h
        half.h
        u64.h
        rng.cpp rng.h
        ies.cpp ies.h
        scattering.cpp scattering.h
        bluenoise.cpp bluenoise.h
        sobolmatrices.cpp sobolmatrices.h
        complex.h
        medium_tracker.cpp medium_tracker.h
        atomic.h
        progress_bar.cpp progress_bar.h)
target_link_libraries(luisa-render-util PUBLIC
        luisa::compute
        luisa-render-include
        luisa-render-ext)
set_target_properties(luisa-render-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
