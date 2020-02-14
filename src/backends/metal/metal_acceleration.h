//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#ifndef __OBJC__
#error This file should only be used in Objective-C/C++ sources.
#endif

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <core/acceleration.h>

namespace luisa::metal {

class MetalAcceleration : public Acceleration {

private:
    MPSAccelerationStructureGroup *_group;
    MPSInstanceAccelerationStructure *_structure;
    MPSRayIntersector *_nearest_intersector;
    MPSRayIntersector *_any_intersector;

public:
    MetalAcceleration(MPSAccelerationStructureGroup *group, MPSInstanceAccelerationStructure *structure, MPSRayIntersector *nearest_its, MPSRayIntersector *any_its) noexcept
        : _group{group}, _structure{structure}, _nearest_intersector{nearest_its}, _any_intersector{any_its} {}
    void refit(KernelDispatcher &dispatch) override;
    void trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<AnyHit> its_buffer, BufferView<uint> ray_count_buffer) override;
    void trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<ClosestHit> its_buffer, BufferView<uint> ray_count_buffer) override;
};

}
