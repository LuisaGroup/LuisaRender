add_library(luisa-render-util SHARED
        sampling.cpp sampling.h
        frame.cpp frame.h
        imageio.cpp imageio.h
        xform.cpp xform.h
        spectrum.cpp spectrum.h colorspace.cpp colorspace.h)
target_link_libraries(luisa-render-util PUBLIC
        luisa::compute
        luisa-render-include
        luisa-render-ext)
set_target_properties(luisa-render-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
