//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/film.h>

namespace luisa {

class RGBFilm : public Film {

private:
    std::unique_ptr<Buffer> _framebuffer;
    std::unique_ptr<Kernel> _postprocess_kernel;

public:
    RGBFilm(Device *device, const ParameterSet &parameters);
    void postprocess(KernelDispatcher &dispatch) override;
    void save(const std::filesystem::path &filename) override;
    
};

}
