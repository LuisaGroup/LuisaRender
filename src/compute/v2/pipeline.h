//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <vector>

#include <compute/v2/buffer.h>
#include <compute/v2/kernel.h>
#include <compute/v2/dispatcher.h>

namespace luisa::compute {

class PipelineStage {

private:
    Kernel *_kernel{nullptr};
    std::vector<Buffer *> _readonly_buffers;
    std::vector<Buffer *> _readwrite_buffers;

public:
    void add_readonly_buffer(Buffer *buffer) noexcept { _readonly_buffers.emplace_back(buffer); }
    void add_readwrite_buffer(Buffer *buffer) noexcept { _readwrite_buffers.emplace_back(buffer); }
    [[nodiscard]] const std::vector<Buffer *> &readonly_buffers() const noexcept { return _readonly_buffers; }
    [[nodiscard]] const std::vector<Buffer *> &readwrite_buffers() const noexcept { return _readwrite_buffers; }
    [[nodiscard]] Kernel *kernel() const noexcept { return _kernel; }
};

class Pipeline {

private:


};

}
