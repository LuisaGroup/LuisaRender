//
// Created by Mike Smith on 2023/5/18.
//

#include <core/logging.h>
#include <util/command_buffer.h>

namespace luisa::render {

CommandBuffer::CommandBuffer(Stream *stream) noexcept
    : _stream{stream} {}

CommandBuffer::~CommandBuffer() noexcept {
    LUISA_ASSERT(_list.empty(),
                 "Command buffer not empty when destroyed. "
                 "Did you forget to commit?");
}

}// namespace luisa::render
