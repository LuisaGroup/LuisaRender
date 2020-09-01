//
// Created by Mike on 9/1/2020.
//

#include "cuda_texture.h"
#include "cuda_buffer.h"
#include "cuda_dispatcher.h"

namespace luisa::cuda {

void CudaTexture::_copy_from(Dispatcher &dispatcher, Buffer *buffer, size_t offset) {
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_MEMCPY2D memcpy_desc{};
    memcpy_desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy_desc.srcDevice = dynamic_cast<CudaBuffer *>(buffer)->handle();
    memcpy_desc.srcXInBytes = 0;
    memcpy_desc.srcY = 0;
    memcpy_desc.srcPitch = pitch_byte_size();
    memcpy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_desc.dstArray = _array_handle;
    memcpy_desc.dstXInBytes = 0;
    memcpy_desc.dstY = 0;
    memcpy_desc.WidthInBytes = pitch_byte_size();
    memcpy_desc.Height = height();
    CUDA_CHECK(cuMemcpy2DAsync(&memcpy_desc, stream));
}

void CudaTexture::_copy_to(Dispatcher &dispatcher, Buffer *buffer, size_t offset) {
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_MEMCPY2D memcpy_desc{};
    memcpy_desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_desc.srcArray = _array_handle;
    memcpy_desc.srcXInBytes = 0;
    memcpy_desc.srcY = 0;
    memcpy_desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy_desc.dstDevice = dynamic_cast<CudaBuffer *>(buffer)->handle();
    memcpy_desc.dstXInBytes = 0;
    memcpy_desc.dstY = 0;
    memcpy_desc.dstPitch = pitch_byte_size();
    memcpy_desc.WidthInBytes = pitch_byte_size();
    memcpy_desc.Height = height();
    CUDA_CHECK(cuMemcpy2DAsync(&memcpy_desc, stream));
}

void CudaTexture::_copy_to(Dispatcher &dispatcher, Texture *texture) {
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_MEMCPY2D memcpy_desc{};
    memcpy_desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_desc.srcArray = _array_handle;
    memcpy_desc.srcXInBytes = 0;
    memcpy_desc.srcY = 0;
    memcpy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_desc.dstArray = dynamic_cast<CudaTexture *>(texture)->array_handle();
    memcpy_desc.dstXInBytes = 0;
    memcpy_desc.dstY = 0;
    memcpy_desc.WidthInBytes = pitch_byte_size();
    memcpy_desc.Height = height();
    CUDA_CHECK(cuMemcpy2DAsync(&memcpy_desc, stream));
}

void CudaTexture::_copy_from(Dispatcher &dispatcher, const void *data) {
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    auto cache = _cache.obtain();
    std::memmove(cache, data, byte_size());
    CUDA_CHECK(cuMemcpyHtoAAsync(_array_handle, 0, cache, byte_size(), stream));
    dispatcher.when_completed([this, cache] { _cache.recycle(cache); });
}

void CudaTexture::_copy_to(Dispatcher &dispatcher, void *data) {
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_MEMCPY2D memcpy_desc{};
    memcpy_desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_desc.srcArray = _array_handle;
    memcpy_desc.srcXInBytes = 0;
    memcpy_desc.srcY = 0;
    memcpy_desc.dstMemoryType = CU_MEMORYTYPE_HOST;
    memcpy_desc.dstHost = data;
    memcpy_desc.dstXInBytes = 0;
    memcpy_desc.dstY = 0;
    memcpy_desc.dstPitch = pitch_byte_size();
    memcpy_desc.WidthInBytes = pitch_byte_size();
    memcpy_desc.Height = height();
    CUDA_CHECK(cuMemcpy2DAsync(&memcpy_desc, stream));
}

}// namespace luisa::cuda
