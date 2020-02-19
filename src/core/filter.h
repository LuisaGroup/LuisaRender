//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "data_types.h"
#include "node.h"
#include "kernel.h"
#include "ray.h"
#include "parser.h"
#include "viewport.h"
#include "mathematics.h"

namespace luisa {

class Film;

class Filter : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Filter);

protected:
    float _radius;
    
    [[nodiscard]] auto _filter_viewport(Viewport film_viewport, Viewport tile_viewport) const {
        auto x_min = max(film_viewport.origin.x, static_cast<uint>(max(0.0f, tile_viewport.origin.x - _radius + 0.5f)));
        auto x_max = min(film_viewport.origin.x + film_viewport.size.x - 1u, static_cast<uint>(static_cast<float>(tile_viewport.origin.x + tile_viewport.size.x) + _radius - 0.5f));
        auto y_min = max(film_viewport.origin.y, static_cast<uint>(max(0.0f, tile_viewport.origin.y - _radius + 0.5f)));
        auto y_max = min(film_viewport.origin.y + film_viewport.size.y - 1u, static_cast<uint>(static_cast<float>(tile_viewport.origin.y + tile_viewport.size.y) + _radius - 0.5f));
        return Viewport{make_uint2(x_min, y_min), make_uint2(x_max - x_min + 1u, y_max - y_min + 1u)};
    }

public:
    Filter(Device *device, const ParameterSet &parameters)
        : Node{device}, _radius{parameters["radius"].parse_float_or_default(1.0f)} {}
    
    virtual void apply_and_accumulate(KernelDispatcher &dispatch,
                                      uint2 film_resolution,
                                      Viewport film_viewport,
                                      Viewport tile_viewport,
                                      BufferView<float2> pixel_buffer,
                                      BufferView<float3> color_buffer,
                                      BufferView<float4> accumulation_buffer) = 0;
    [[nodiscard]] float radius() const noexcept { return _radius; }
};
    
}
