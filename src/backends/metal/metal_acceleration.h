//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <render/acceleration.h>

namespace luisa::metal {

using compute::Buffer;
using compute::BufferView;
using compute::Dispatcher;

using render::Ray;
using render::Acceleration;

class MetalAcceleration : public Acceleration {

private:
    MPSAccelerationStructureGroup *_group;
    MPSInstanceAccelerationStructure *_structure;
    MPSRayIntersector *_nearest_intersector;
    MPSRayIntersector *_any_intersector;

public:
    MetalAcceleration(MPSAccelerationStructureGroup *group, MPSInstanceAccelerationStructure *structure, MPSRayIntersector *nearest_its, MPSRayIntersector *any_its) noexcept
        : _group{group}, _structure{structure}, _nearest_intersector{nearest_its}, _any_intersector{any_its} {}
        
    ~MetalAcceleration() noexcept override = default;
    
    void refit(Dispatcher &dispatch) override;
    void intersect_any(Dispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<AnyHit> its_buffer, BufferView<uint> ray_count_buffer) override;
    void intersect_closest(Dispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<ClosestHit> its_buffer, BufferView<uint> ray_count_buffer) override;
};

}
