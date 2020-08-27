message(STATUS "Building with NVIDIA CUDA support")

file(GLOB SOURCE_FILES *.h *.cpp)
luisa_render_add_backend(cuda SOURCES ${SOURCE_FILES})
target_include_directories(cuda PRIVATE ${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::cudart_static)
