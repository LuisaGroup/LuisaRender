function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_link_libraries(${name} PRIVATE luisa::render)
    install(TARGETS ${name}
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

luisa_render_add_application(luisa-render-cli SOURCES cli.cpp)
luisa_render_add_application(luisa-render-export SOURCES export.cpp)
