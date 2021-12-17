//
// Created by Mike on 2021/12/17.
//

#pragma once

#include <luisa-compute.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelBuildHint;
using compute::BindlessArray;
using compute::Buffer;
using compute::Device;
using compute::Image;
using compute::Mesh;
using compute::PixelStorage;
using compute::Resource;
using compute::Volume;
using compute::Callable;

class PipelineData {

public:
    struct alignas(8) MeshData {
        uint position_buffer_id;
        uint normal_buffer_id;
        uint uv_buffer_id;
        uint triangle_buffer_id;
        uint triangle_count;
        uint area_cdf_buffer;
    };

private:




};

}
