//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#ifndef __OBJC__
#error This file should only be used in Objective-C/C++ sources.
#endif

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <core/acceleration.h>

class MetalAcceleration : public Acceleration {

private:
    MPSTriangleAccelerationStructure *_structure;
    MPSRayIntersector *_nearest_intersector;
    MPSRayIntersector *_any_intersector;

public:
    MetalAcceleration(MPSTriangleAccelerationStructure *structure, MPSRayIntersector *nearest_its, MPSRayIntersector *any_its) noexcept
        : _structure{structure}, _nearest_intersector{nearest_its}, _any_intersector{any_its} {}
    
    void trace_any(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, size_t ray_count) override;
    void trace_nearest(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, size_t ray_count) override;
    void trace_any(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, Buffer &ray_count_buffer, size_t ray_count_buffer_offset) override;
    void trace_nearest(KernelDispatcher &dispatch, Buffer &ray_buffer, Buffer &intersection_buffer, Buffer &ray_count_buffer, size_t ray_count_buffer_offset) override;
    void trace_any(KernelDispatcher &dispatch,
                   Buffer &ray_buffer,
                   Buffer &ray_index_buffer,
                   Buffer &intersection_buffer,
                   Buffer &ray_count_buffer,
                   size_t ray_count_buffer_offset) override;
    void trace_nearest(KernelDispatcher &dispatch,
                       Buffer &ray_buffer,
                       Buffer &ray_index_buffer,
                       Buffer &intersection_buffer,
                       Buffer &ray_count_buffer,
                       size_t ray_count_buffer_offset) override;
};


