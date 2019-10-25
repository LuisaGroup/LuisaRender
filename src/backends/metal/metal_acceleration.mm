//
// Created by Mike Smith on 2019/10/25.
//

#include "metal_acceleration.h"
#import "metal_kernel.h"
#import "metal_buffer.h"

void MetalAcceleration::trace_any(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, size_t ray_count) {
    [_any_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                       intersectionType:MPSIntersectionTypeAny
                                              rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).buffer()
                                        rayBufferOffset:0u
                                     intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).buffer()
                               intersectionBufferOffset:0u
                                               rayCount:ray_count
                                  accelerationStructure:_structure];
}

void MetalAcceleration::trace_nearest(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, size_t ray_count) {
    [_nearest_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                       intersectionType:MPSIntersectionTypeNearest
                                              rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).buffer()
                                        rayBufferOffset:0u
                                     intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).buffer()
                               intersectionBufferOffset:0u
                                               rayCount:ray_count
                                  accelerationStructure:_structure];
}
