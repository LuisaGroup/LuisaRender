//
// Created by Mike Smith on 2020/8/14.
//

#pragma once

#import <Metal/Metal.h>
#import <compute/dispatcher.h>

namespace luisa::metal {

using compute::Dispatcher;

class MetalDispatcher : public Dispatcher {

private:
    id<MTLCommandBuffer> _handle{nullptr};

protected:
    void _synchronize() override { [_handle waitUntilCompleted]; }
    void _commit() override { [_handle commit]; }

public:
    explicit MetalDispatcher() noexcept = default;
    [[nodiscard]] id<MTLCommandBuffer> handle() const noexcept { return _handle; }
    void commit() override { if (_handle != nullptr) { Dispatcher::commit(); }}
    void synchronize() override { if (_handle != nullptr) { Dispatcher::synchronize(); }}
    void reset(id<MTLCommandBuffer> handle = nullptr) noexcept {
        _callbacks.clear();
        _handle = handle;
    }
};

}
