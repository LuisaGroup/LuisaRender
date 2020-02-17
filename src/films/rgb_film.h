//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/colorspaces.h>

namespace luisa::rgb_film {

LUISA_DEVICE_CALLABLE inline void postprocess(
    LUISA_DEVICE_SPACE const float4 *accumulation_buffer,
    LUISA_DEVICE_SPACE float4 *framebuffer,
    uint pixel_count,
    uint tid) noexcept {
    
    if (tid < pixel_count) {
        auto f = accumulation_buffer[tid];
        framebuffer[tid] = make_float4(ACEScg2XYZ(XYZ2RGB(make_float3(f) / f.a)), 1.0f);
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

LUISA_REGISTER_NODE_CREATOR("RGB", RGBFilm);

}

#endif
