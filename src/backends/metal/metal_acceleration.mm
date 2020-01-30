//
// Created by Mike Smith on 2019/10/25.
//

#import "metal_acceleration.h"
#import "metal_kernel.h"
#import "metal_buffer.h"

namespace luisa::metal {

void MetalAcceleration::trace_any(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, size_t ray_count) {
    [_any_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                       intersectionType:MPSIntersectionTypeAny
                                              rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).handle()
                                        rayBufferOffset:0u
                                     intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).handle()
                               intersectionBufferOffset:0u
                                               rayCount:ray_count
                                  accelerationStructure:_structure];
}

void MetalAcceleration::trace_nearest(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, size_t ray_count) {
    [_nearest_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                           intersectionType:MPSIntersectionTypeNearest
                                                  rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).handle()
                                            rayBufferOffset:0u
                                         intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).handle()
                                   intersectionBufferOffset:0u
                                                   rayCount:ray_count
                                      accelerationStructure:_structure];
}

void MetalAcceleration::trace_any(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, Buffer &ray_count_buffer, size_t ray_count_buffer_offset) {
    [_any_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                       intersectionType:MPSIntersectionTypeAny
                                              rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).handle()
                                        rayBufferOffset:0u
                                     intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).handle()
                               intersectionBufferOffset:0u
                                         rayCountBuffer:dynamic_cast<MetalBuffer &>(ray_count_buffer).handle()
                                   rayCountBufferOffset:ray_count_buffer_offset
                                  accelerationStructure:_structure];
}

void MetalAcceleration::trace_nearest(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, Buffer &ray_count_buffer, size_t ray_count_buffer_offset) {
    [_nearest_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                           intersectionType:MPSIntersectionTypeNearest
                                                  rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).handle()
                                            rayBufferOffset:0u
                                         intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).handle()
                                   intersectionBufferOffset:0u
                                             rayCountBuffer:dynamic_cast<MetalBuffer &>(ray_count_buffer).handle()
                                       rayCountBufferOffset:ray_count_buffer_offset
                                      accelerationStructure:_structure];
}

void MetalAcceleration::trace_any(KernelDispatcher &dispatch,
                                  Buffer &ray_buffer,
                                  Buffer &ray_index_buffer,
                                  Buffer &intersection_buffer,
                                  Buffer &ray_count_buffer,
                                  size_t ray_count_buffer_offset) {
    
    [_any_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                       intersectionType:MPSIntersectionTypeAny
                                              rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).handle()
                                        rayBufferOffset:0u
                                         rayIndexBuffer:dynamic_cast<MetalBuffer &>(ray_index_buffer).handle()
                                   rayIndexBufferOffset:0u
                                     intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).handle()
                               intersectionBufferOffset:0u
                                    rayIndexCountBuffer:dynamic_cast<MetalBuffer &>(ray_count_buffer).handle()
                              rayIndexCountBufferOffset:ray_count_buffer_offset
                                  accelerationStructure:_structure];
    
}

void MetalAcceleration::trace_nearest(KernelDispatcher &dispatch,
                                      Buffer &ray_buffer,
                                      Buffer &ray_index_buffer,
                                      Buffer &intersection_buffer,
                                      Buffer &ray_count_buffer,
                                      size_t ray_count_buffer_offset) {
    
    [_nearest_intersector encodeIntersectionToCommandBuffer:dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer()
                                           intersectionType:MPSIntersectionTypeNearest
                                                  rayBuffer:dynamic_cast<MetalBuffer &>(ray_buffer).handle()
                                            rayBufferOffset:0u
                                             rayIndexBuffer:dynamic_cast<MetalBuffer &>(ray_index_buffer).handle()
                                       rayIndexBufferOffset:0u
                                         intersectionBuffer:dynamic_cast<MetalBuffer &>(intersection_buffer).handle()
                                   intersectionBufferOffset:0u
                                        rayIndexCountBuffer:dynamic_cast<MetalBuffer &>(ray_count_buffer).handle()
                                  rayIndexCountBufferOffset:ray_count_buffer_offset
                                      accelerationStructure:_structure];
    
}

}
