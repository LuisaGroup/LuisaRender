set(LUISA_RENDER_SDL_SOURCES
        scene_desc.cpp scene_desc.h
        scene_node_desc.cpp scene_node_desc.h
        scene_node_tag.h
        scene_parser.cpp scene_parser.h scene_parser_json.cpp scene_parser_json.h scene_node_tag.cpp)

add_library(luisa-render-sdl SHARED ${LUISA_RENDER_SDL_SOURCES})
target_link_libraries(luisa-render-sdl PUBLIC
        luisa::compute
        luisa-render-include
        luisa-render-ext
        luisa-render-util)
set_target_properties(luisa-render-sdl PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_RENDER_ENABLE_UNITY_BUILD})

install(TARGETS luisa-render-sdl
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
