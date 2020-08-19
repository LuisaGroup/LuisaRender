//
// Created by Mike Smith on 2020/8/19.
//

#pragma once

#include <compute/device.h>
#include <compute/dsl.h>
#include <compute/kernel.h>

#include "box_blur.h"

using luisa::compute::Device;
using luisa::compute::Kernel;
using luisa::compute::Texture;
using luisa::compute::dsl::Function;

class ResidualVariance {

private:
    int _width;
    int _height;
    std::unique_ptr<Texture> _temp;
    std::unique_ptr<Kernel> _diff_kernel;

public:
    ResidualVariance(Device &device, float sigma, Texture &color_a, Texture &color_b, Texture &output);

};


