message(STATUS "Building with NVIDIA CUDA support")

file(GLOB SOURCE_FILES *.h *.cpp)
luisa_render_add_backend(cuda SOURCES ${SOURCE_FILES})
target_include_directories(luisa-backend-cuda PRIVATE ${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED VERSION 7)
target_link_libraries(luisa-backend-cuda PRIVATE jitify_config CUDA::nvrtc CUDA::cuda_driver)

if (OptiX_FOUND)
    target_include_directories(luisa-backend-cuda PRIVATE ${OptiX_INCLUDE})
    target_compile_definitions(luisa-backend-cuda PRIVATE
                               LUISA_OPTIX_AVAILABLE=1
                               LUISA_OPTIX_INCLUDE_DIR="${OptiX_INCLUDE}")
    enable_language(CUDA)
    add_library(luisa-optix-adapter OBJECT optix_adapter.h optix_adapter.cu)
    set_target_properties(luisa-optix-adapter PROPERTIES CUDA_PTX_COMPILATION ON)
    target_include_directories(luisa-optix-adapter PRIVATE ${OPTIX_INCLUDE})
endif ()
