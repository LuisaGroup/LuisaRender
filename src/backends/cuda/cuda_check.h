//
// Created by Mike on 8/27/2020.
//

#pragma once

#include <core/logging.h>

#define NVRTC_CHECK(x)                                                                                \
    [&] {                                                                                             \
        nvrtcResult result = x;                                                                       \
        LUISA_EXCEPTION_IF_NOT(                                                                       \
            result == NVRTC_SUCCESS, "MVRTC call [ " #x " ] failed: ", nvrtcGetErrorString(result));  \
    }()

#define CUDA_CHECK(x)                                               \
    [&] {                                                           \
        CUresult result = x;                                        \
        if (result != CUDA_SUCCESS) {                               \
            const char *msg;                                        \
            cuGetErrorName(result, &msg);                           \
            LUISA_EXCEPTION("CUDA call [ " #x " ] failed: ", msg);  \
        }                                                           \
    }()
