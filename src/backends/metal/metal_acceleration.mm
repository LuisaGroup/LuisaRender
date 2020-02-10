//
// Created by Mike Smith on 2019/10/25.
//

#import "metal_acceleration.h"
#import "metal_kernel.h"
#import "metal_buffer.h"

namespace luisa::metal {

void MetalAcceleration::refit(KernelDispatcher &dispatch) {
    [_structure encodeRefitToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()];
}

void MetalAcceleration::trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<Intersection> its_buffer, BufferView<uint> ray_count_buffer) {
    [_any_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                       intersectionType:MPSIntersectionTypeAny
                                              rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer.buffer()).handle()
                                        rayBufferOffset:ray_buffer.byte_offset()
                                     intersectionBuffer:dynamic_cast<MetalBuffer &>(its_buffer.buffer()).handle()
                               intersectionBufferOffset:its_buffer.byte_offset()
                                         rayCountBuffer:dynamic_cast<MetalBuffer &>(ray_count_buffer.buffer()).handle()
                                   rayCountBufferOffset:ray_count_buffer.byte_offset()
                                  accelerationStructure:_structure];
}

void MetalAcceleration::trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<Intersection> its_buffer, BufferView<uint> ray_count_buffer) {
    [_nearest_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                           intersectionType:MPSIntersectionTypeNearest
                                                  rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer.buffer()).handle()
                                            rayBufferOffset:ray_buffer.byte_offset()
                                         intersectionBuffer:dynamic_cast<MetalBuffer &>(its_buffer.buffer()).handle()
                                   intersectionBufferOffset:its_buffer.byte_offset()
                                             rayCountBuffer:dynamic_cast<MetalBuffer &>(ray_count_buffer.buffer()).handle()
                                       rayCountBufferOffset:ray_count_buffer.byte_offset()
                                      accelerationStructure:_structure];
}

}
