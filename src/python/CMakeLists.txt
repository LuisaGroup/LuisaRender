set(LUISARENDER_PYTHON_SOURCES
        lrapi.cpp)

#find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
#find_package(pybind11 CONFIG REQUIRED)

#python_add_library(_lrapi MODULE ${LUISARENDER_PYTHON_SOURCES} WITH_SOABI)
pybind11_add_module(_lrapi ${LUISARENDER_PYTHON_SOURCES})

if (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(_lrapi PRIVATE -Wa,-mbig-obj)
endif ()
target_link_libraries(_lrapi PRIVATE luisa::render)
target_link_libraries(_lrapi PRIVATE pybind11::headers)

if (SKBUILD)
    install(DIRECTORY luisarender
        DESTINATION ${SKBUILD_PLATLIB_DIR})
    install(TARGETS _lrapi
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()
