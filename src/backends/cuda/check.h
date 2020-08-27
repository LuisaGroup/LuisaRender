//
// Created by Mike on 8/27/2020.
//

#pragma once

#include <core/logging.h>

#define OPTIX_CHECK(call)                                                       \
    [&] {                                                                       \
        if (auto res = call; res != OPTIX_SUCCESS) {                            \
            LUISA_ERROR("OptiX call [ ", #call, " ] ",                          \
                        "failed with error: ", optixGetErrorString(res), ": ",  \
                        __FILE__, ":", __LINE__);                               \
        }                                                                       \
    }()

#define CUDA_CHECK(call)                                                       \
    [&] {                                                                      \
        if (auto res = call; res != 0) {                                       \
            LUISA_ERROR("CUDA call [ ", #call, " ] ",                          \
                        "failed with error: ", cudaGetErrorString(res), ": ",  \
                        __FILE__, ":", __LINE__);                              \
        }                                                                      \
    }()
