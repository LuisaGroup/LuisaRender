add_library(luisa-render-util SHARED
        sampling.cpp sampling.h
        frame.cpp frame.h
        imageio.cpp imageio.h
        xform.cpp xform.h
        spectrum.cpp spectrum.h
        spectrum_cie_xyz.inl.h
        spectrum_cie_illum_d65.inl.h
        colorspace.cpp colorspace.h
        srgb2spec.cpp
        half.h
        u64.h
        rng.cpp rng.h
        ies.cpp ies.h
        scattering.cpp scattering.h
        bluenoise.cpp bluenoise.h
        sobolmatrices.cpp sobolmatrices.h
        complex.h)
target_link_libraries(luisa-render-util PUBLIC
        luisa::compute
        luisa-render-include
        luisa-render-ext)
set_target_properties(luisa-render-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
