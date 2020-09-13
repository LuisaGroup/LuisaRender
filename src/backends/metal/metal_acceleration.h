//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <compute/acceleration.h>

namespace luisa::metal {

using compute::Ray;
using compute::AnyHit;
using compute::ClosestHit;
using compute::BufferView;
using compute::Acceleration;

class MetalAcceleration : public Acceleration {

private:
    MPSInstanceAccelerationStructure *_as;
    MPSRayIntersector *_closest_intersector;
    MPSRayIntersector *_any_intersector;

private:
    void _refit(compute::Dispatcher &dispatch) override;
    
    void _intersect_any(
        compute::Dispatcher &dispatch,
        const BufferView<Ray> &ray_buffer,
        const BufferView<AnyHit> &hit_buffer) const override;
    
    void _intersect_closest(compute::Dispatcher &dispatch,
                            const BufferView<Ray> &ray_buffer,
                            const BufferView<ClosestHit> &hit_buffer) const override;

public:
    MetalAcceleration(MPSInstanceAccelerationStructure *as, MPSRayIntersector *closest_its, MPSRayIntersector *any_its) noexcept
        : _as{as}, _closest_intersector{closest_its}, _any_intersector{any_its} {}
};

}
