//
// Created by Mike Smith on 2019/10/25.
//

#import "metal_acceleration.h"
#import "metal_dispatcher.h"
#import "metal_buffer.h"

namespace luisa::metal {

void MetalAcceleration::refit(Dispatcher &dispatch) {
    [_structure encodeRefitToCommandBuffer:dynamic_cast<MetalDispatcher &>(dispatch).handle()];
}

void MetalAcceleration::intersect_any(Dispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<AnyHit> its_buffer, BufferView<uint> ray_count_buffer) {
    [_any_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalDispatcher &>(dispatch).handle()
                                       intersectionType:MPSIntersectionTypeAny
                                              rayBuffer:dynamic_cast<MetalBuffer *>(ray_buffer.buffer())->handle()
                                        rayBufferOffset:ray_buffer.byte_offset()
                                     intersectionBuffer:dynamic_cast<MetalBuffer *>(its_buffer.buffer())->handle()
                               intersectionBufferOffset:its_buffer.byte_offset()
                                         rayCountBuffer:dynamic_cast<MetalBuffer *>(ray_count_buffer.buffer())->handle()
                                   rayCountBufferOffset:ray_count_buffer.byte_offset()
                                  accelerationStructure:_structure];
}

void MetalAcceleration::intersect_closest(Dispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<ClosestHit> its_buffer, BufferView<uint> ray_count_buffer) {
    [_nearest_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalDispatcher &>(dispatch).handle()
                                           intersectionType:MPSIntersectionTypeNearest
                                                  rayBuffer:dynamic_cast<MetalBuffer *>(ray_buffer.buffer())->handle()
                                            rayBufferOffset:ray_buffer.byte_offset()
                                         intersectionBuffer:dynamic_cast<MetalBuffer *>(its_buffer.buffer())->handle()
                                   intersectionBufferOffset:its_buffer.byte_offset()
                                             rayCountBuffer:dynamic_cast<MetalBuffer *>(ray_count_buffer.buffer())->handle()
                                       rayCountBufferOffset:ray_count_buffer.byte_offset()
                                      accelerationStructure:_structure];
}

}
