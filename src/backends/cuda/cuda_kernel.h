//
// Created by Mike on 9/1/2020.
//

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>

#include <cuda.h>

#include <compute/function.h>
#include <compute/kernel.h>
#include <core/logging.h>
#include <core/platform.h>

#include "cuda_check.h"

namespace luisa::cuda {

using luisa::compute::Kernel;
using luisa::compute::dsl::Variable;

class CudaKernel : public Kernel {

private:
    CUfunction _handle;
    std::vector<std::byte> _arguments{};

protected:
    void _dispatch(compute::Dispatcher &dispatcher, uint2 blocks, uint2 block_size) override;

public:
    CudaKernel(CUfunction handle, std::vector<Kernel::Resource> resources, std::vector<Kernel::Uniform> uniforms) noexcept;
};

}// namespace luisa::cuda
