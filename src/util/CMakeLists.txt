add_library(luisa-render-util SHARED
        sampling.cpp sampling.h)
target_link_libraries(luisa-render-util PUBLIC
        luisa::compute luisa-render-include)
