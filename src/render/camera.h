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

using compute::Pipeline;
using compute::BufferView;
using compute::TextureView;
using compute::Ray;
using compute::dsl::Expr;

class Camera : public Plugin {

private:
    std::shared_ptr<Film> _film;
    std::shared_ptr<Filter> _filter;
    std::shared_ptr<Transform> _transform;
    
    BufferView<float2> _pixel_position_buffer;
    BufferView<Ray> _camera_ray_buffer;
    BufferView<float3> _throughput_buffer;
    BufferView<float> _filter_weight_buffer;

private:
    virtual void _generate_rays(Pipeline &pipeline,
                                float4x4 camera_to_world,
                                Sampler &sampler,
                                const BufferView<float2> &pixel_positions,
                                BufferView<Ray> &rays,
                                BufferView<float3> &throughputs) = 0;
    
    void _generate_pixel_positions_without_filter(Pipeline &pipeline, Sampler &sampler, BufferView<float2> &pixel_position_buffer, BufferView<float> &filter_weight_buffer);

public:
    Camera(Device *d, const ParameterSet &params)
        : Plugin{d, params},
          _film{params["film"].parse_or_null<Film>()},
          _filter{params["filter"].parse_or_null<Filter>()},
          _transform{params["filter"].parse_or_null<Transform>()} {
        
        auto pixel_count = _film->resolution().x * _film->resolution().y;
        _pixel_position_buffer = device()->allocate_buffer<float2>(pixel_count);
        _camera_ray_buffer = device()->allocate_buffer<Ray>(pixel_count);
        _throughput_buffer = device()->allocate_buffer<float3>(pixel_count);
    }
    
    [[nodiscard]] Film *film() const noexcept { return _film.get(); }
    [[nodiscard]] Filter *filter() const noexcept { return _filter.get(); }
    [[nodiscard]] Transform *transform() const noexcept { return _transform.get(); }
    
    [[nodiscard]] const BufferView<float2> &pixel_position_buffer() const noexcept { return _pixel_position_buffer; }
    [[nodiscard]] BufferView<Ray> &ray_buffer() noexcept { return _camera_ray_buffer; }
    [[nodiscard]] BufferView<float3> &throughput_buffer() noexcept { return _throughput_buffer; }
    
    [[nodiscard]] auto generate_rays(float time, Sampler &sampler) {
        return [this, time, &sampler](Pipeline &pipeline) {
            if (_filter == nullptr) {
                _generate_pixel_positions_without_filter(pipeline, sampler, _pixel_position_buffer, _filter_weight_buffer);
            } else {
                pipeline << _filter->importance_sample_pixel_positions(sampler, _pixel_position_buffer, _filter_weight_buffer);
            }
            auto camera_to_world = _transform == nullptr ? make_float4x4(1.0f) : _transform->matrix(time);
            _generate_rays(pipeline, camera_to_world, sampler, _pixel_position_buffer, _camera_ray_buffer, _throughput_buffer);
        };
    }
};

}
