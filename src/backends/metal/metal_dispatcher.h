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

private:
    id<MTLCommandBuffer> _handle{nullptr};

public:
    explicit MetalDispatcher() noexcept = default;
    ~MetalDispatcher() noexcept { [_handle waitUntilCompleted]; }
    [[nodiscard]] id<MTLCommandBuffer> handle() const noexcept { return _handle; }
    void reset(id<MTLCommandBuffer> handle = nullptr) noexcept {
        _callbacks.clear();
        _handle = handle;
    }
    
    void wait() override { [_handle waitUntilCompleted]; }
    void commit() override {
        [_handle addCompletedHandler:^(id<MTLCommandBuffer>) { for (auto &&callback : _callbacks) { callback(); } }];
        [_handle commit];
    }
};

}
