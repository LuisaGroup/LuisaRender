//
// Created by Mike Smith on 2020/8/14.
//

#pragma once

#import <Metal/Metal.h>
#import <compute/dispatcher.h>

namespace luisa::metal {

class MetalDevice;

using compute::Dispatcher;

class MetalDispatcher : public Dispatcher {

public:
    friend class MetalDevice;

private:
    id<MTLCommandBuffer> _handle{nullptr};

protected:
    void _do_synchronize() override { [_handle waitUntilCompleted]; }
    void _do_commit() override { [_handle commit]; }

public:
    explicit MetalDispatcher() noexcept = default;
    [[nodiscard]] id<MTLCommandBuffer> handle() const noexcept { return _handle; }
    void reset(id<MTLCommandBuffer> handle = nullptr) noexcept {
        _callbacks.clear();
        _handle = handle;
    }
};

}
