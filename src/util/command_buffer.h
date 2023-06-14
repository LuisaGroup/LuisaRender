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
    explicit CommandBuffer(Stream *stream) noexcept;
    ~CommandBuffer() noexcept;

    [[nodiscard]] auto stream() const noexcept { return _stream; }

    template<typename T>
    CommandBuffer &operator<<(T &&cmd) noexcept {
        if constexpr (requires { _list << std::forward<T>(cmd); }) {
            _list << std::forward<T>(cmd);
        } else {
            *_stream << _list.commit()
                     << std::forward<T>(cmd);
        }
        return *this;
    }

    CommandBuffer &operator<<(compute::Stream::Commit) noexcept {
        if (!_list.empty()) { *_stream << _list.commit(); }
        return *this;
    }

    CommandBuffer &operator<<(compute::Stream::Synchronize) noexcept {
        if (!_list.empty()) { *_stream << _list.commit(); }
        _stream->synchronize();
        return *this;
    }

    template<typename... T>
    CommandBuffer &operator<<(std::tuple<T...> cmds) noexcept {
        std::apply(
            [&]<typename... Cmd>(Cmd &&...cmd) noexcept {
                ((*this) << ... << std::forward<Cmd>(cmd));
            },
            std::move(cmds));
        return *this;
    }
};

}// namespace luisa::render