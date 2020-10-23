message(STATUS "Building with NVIDIA CUDA support")

file(GLOB SOURCE_FILES *.h *.cpp)
luisa_render_add_backend(cuda SOURCES ${SOURCE_FILES})
target_include_directories(luisa-backend-cuda PRIVATE ${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
find_package(OptiX 6.5)
target_link_libraries(luisa-backend-cuda PRIVATE jitify_config CUDA::nvrtc CUDA::cuda_driver)

if (OptiX_FOUND)
    message(STATUS "Building with NVIDIA OptiX support")
    target_link_libraries(luisa-backend-cuda PRIVATE optix_prime)
    target_compile_definitions(luisa-backend-cuda PRIVATE LUISA_OPTIX_AVAILABLE=1)
    if (optix_prime_DLL)
        add_custom_command(TARGET luisa-backend-cuda POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy
                           "${optix_prime_DLL}" "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
                           COMMENT "Copying OptiX Prime DLL ${optix_prime_DLL}")
    endif ()
endif ()
