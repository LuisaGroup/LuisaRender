//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/colorspaces.h>
#include <core/mathematics.h>

namespace luisa::film::rgb {

LUISA_DEVICE_CALLABLE inline void postprocess(
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    uint pixel_count,
    uint tid) noexcept {
    
    if (tid < pixel_count) {
        auto f = accumulation_buffer[tid];
        accumulation_buffer[tid] = f.w == 0.0f ? make_float4() : make_float4(XYZ2RGB(ACEScg2XYZ(make_float3(f) / f.w)), 1.0f);
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/film.h>

namespace luisa {

class RGBFilm : public Film {

private:
    std::unique_ptr<Kernel> _postprocess_kernel;

public:
    RGBFilm(Device *device, const ParameterSet &parameters);
    void postprocess(KernelDispatcher &dispatch) override;
    void save(const std::filesystem::path &filename) override;
    
};

}

#endif
