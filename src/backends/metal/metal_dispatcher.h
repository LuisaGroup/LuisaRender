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
    static constexpr auto max_commands_in_single_dispatch = 8u;

private:
    id<MTLCommandBuffer> _handle{nullptr};
    uint _dispatch_count{0u};

protected:
    void _on_dispatch() override {
        if (++_dispatch_count >= max_commands_in_single_dispatch) {
            [_handle enqueue];
            [_handle commit];
            _handle = [[_handle commandQueue] commandBufferWithUnretainedReferences];
            _dispatch_count = 0u;
        }
    }

public:
    explicit MetalDispatcher() noexcept = default;
    ~MetalDispatcher() noexcept override { [_handle waitUntilCompleted]; }
    [[nodiscard]] id<MTLCommandBuffer> handle() const noexcept { return _handle; }
    void reset(id<MTLCommandBuffer> handle = nullptr) noexcept {
        _callbacks = {};
        _handle = handle;
        _dispatch_count = 0u;
    }
    
    void wait() override { [_handle waitUntilCompleted]; }
    void commit() override {
        [_handle addCompletedHandler:^(__weak id<MTLCommandBuffer>) {
            for (auto &&callback : _callbacks) {
                callback();
            }
        }];
        [_handle enqueue];
        [_handle commit];
    }
};

}
