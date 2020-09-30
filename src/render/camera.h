//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <compute/pipeline.h>
#include <render/plugin.h>
#include <render/parser.h>
#include <render/film.h>
#include <render/filter.h>
#include <render/transform.h>
#include <render/sampler.h>

namespace luisa::render {

using compute::Ray;

struct RaySample {
    Ray ray;
    float3 throughput;
};

}

LUISA_STRUCT(luisa::render::RaySample, ray, throughput)

namespace luisa::render {

using compute::Pipeline;
using compute::BufferView;
using compute::TextureView;
using compute::dsl::Expr;
using compute::dsl::Var;

class Camera : public Plugin {

private:
    std::shared_ptr<Film> _film;
    std::shared_ptr<Filter> _filter;
    std::shared_ptr<Transform> _transform;
    
    BufferView<float2> _pixel_position_buffer;
    BufferView<Ray> _camera_ray_buffer;
    BufferView<float3> _throughput_buffer;
    BufferView<float> _pixel_weight_buffer;
    
    float4x4 _camera_to_world;

private:
    [[nodiscard]] virtual bool _requires_lens_samples() const noexcept = 0;
    [[nodiscard]] virtual Expr<RaySample> _generate_rays(Var<float4x4> camera_to_world, Var<float2> u_lens, Var<float2> pixel_positions) = 0;

public:
    Camera(Device *d, const ParameterSet &params)
        : Plugin{d, params},
          _film{params["film"].parse_or_null<Film>()},
          _filter{params["filter"].parse_or_null<Filter>()},
          _transform{params["transform"].parse_or_null<Transform>()} {}
    
    [[nodiscard]] Film *film() const noexcept { return _film.get(); }
    [[nodiscard]] Filter *filter() const noexcept { return _filter.get(); }
    [[nodiscard]] Transform *transform() const noexcept { return _transform.get(); }
    
    [[nodiscard]] const BufferView<float2> &pixel_position_buffer() const noexcept { return _pixel_position_buffer; }
    [[nodiscard]] const BufferView<float> &pixel_weight_buffer() const noexcept { return _pixel_weight_buffer; }
    [[nodiscard]] BufferView<Ray> &ray_buffer() noexcept { return _camera_ray_buffer; }
    [[nodiscard]] BufferView<float3> &throughput_buffer() noexcept { return _throughput_buffer; }
    
    [[nodiscard]] std::function<void(Pipeline &pipeline)> generate_rays(float time, Sampler &sampler);
    [[nodiscard]] bool is_static() const noexcept { return _transform == nullptr || _transform->is_static(); }
};

}
