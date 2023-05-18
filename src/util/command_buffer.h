//
// Created by Mike Smith on 2023/5/18.
//

#pragma once

#include <runtime/command_list.h>
#include <runtime/stream.h>

namespace luisa::render {

using compute::CommandList;
using compute::Stream;

class CommandBuffer {

private:
    Stream *_stream;
    CommandList _list;

public:


};

}// namespace luisa::render