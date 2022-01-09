add_library(luisa-render-util SHARED
        sampling.cpp sampling.h)
target_link_libraries(luisa-render-util PUBLIC
        luisa::compute luisa-render-include)
set_target_properties(luisa-render-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
