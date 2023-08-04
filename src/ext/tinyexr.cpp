#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#if !defined(TINYEXR_USE_MINIZ) || TINYEXR_USE_MINIZ == 0
#include <zlib.h>
#endif

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
