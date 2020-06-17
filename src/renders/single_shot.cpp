//
// Created by Mike Smith on 2020/2/11.
//

#include <vector>
#include <random>
#include <filesystem>

#include <core/render.h>

namespace luisa {

class SingleShot : public Render {

protected:
    float _shutter_open;
    float _shutter_close;
    uint _shutter_samples;
    std::shared_ptr<Camera> _camera;
    std::string _output_path;
    Viewport _viewport{};
    
    void _execute() override;

public:
    SingleShot(Device *device, const ParameterSet &parameter_set);
};

SingleShot::SingleShot(Device *device, const ParameterSet &parameter_set)
    : Render{device, parameter_set},
      _shutter_samples{parameter_set["shutter_samples"].parse_uint_or_default(0u)},
      _camera{parameter_set["camera"].parse<Camera>()},
      _output_path{parameter_set["output"].parse_string()} {
    
    auto viewport = parameter_set["viewport"].parse_uint4_or_default(make_uint4(0u, 0u, _camera->film().resolution()));
    _viewport = {make_uint2(viewport.x, viewport.y), make_uint2(viewport.z, viewport.w)};
    
    auto shutter_duration = parameter_set["shutter_duration"].parse_float2_or_default(make_float2());
    _shutter_open = shutter_duration.x;
    _shutter_close = shutter_duration.y;
    if (_shutter_open > _shutter_close) { std::swap(_shutter_open, _shutter_close); }
    
    auto shapes = parameter_set["shapes"].parse_reference_list<Shape>();
    auto lights = parameter_set["lights"].parse_reference_list<Light>();
    _scene = Scene::create(_device, shapes, lights, _shutter_open);
    
    if (_shutter_samples > _sampler->spp()) {
        LUISA_WARNING("Too many shutter samples, limiting to samples per frame");
        _shutter_samples = _sampler->spp();
    }
    
    if (_shutter_samples == 0u) {
        constexpr auto power_two_le = [](uint x) noexcept {
            for (auto i = 31u; i > 0u; i--) { if (auto v = 1u << i; v <= x) { return v; }}
            return 1u;
        };
        auto film_resolution = _camera->film().resolution();
        _shutter_samples = std::clamp(_sampler->spp(), 1u, power_two_le(std::max(film_resolution.x, film_resolution.y) / 4u));
        LUISA_WARNING("Shutter samples not specified, using heuristic value: ", _shutter_samples);
    }
}

void SingleShot::_execute() {
    
    std::default_random_engine random_engine{std::random_device{}()};
    
    auto spp = _sampler->spp();
    auto &&film = _camera->film();
    
    std::vector<float> time_sample_buckets(_shutter_samples);
    for (auto i = 0u; i < time_sample_buckets.size(); i++) {
        time_sample_buckets[i] = lerp(_shutter_open, _shutter_close, (i + std::uniform_real_distribution{0.0f, 1.0f}(random_engine)) / time_sample_buckets.size());
    }
    std::vector<uint> time_sample_counts(_shutter_samples, 0u);
    for (auto i = 0u; i < spp; i++) { time_sample_counts[std::uniform_int_distribution{0u, _shutter_samples - 1u}(random_engine)]++; }
    
    _sampler->reset_states(film.resolution(), _viewport);
    film.reset_accumulation_buffer(_viewport);
    
    LUISA_INFO("Rendering started");
    for (auto i = 0u; i < _shutter_samples; i++) {
        
        auto time = time_sample_buckets[i];
        
        _camera->update(time);
        _scene->update(time);
        
        for (auto s = 0u; s < time_sample_counts[i]; s++) {
            
            _integrator->prepare_for_frame(_scene.get(), _camera.get(), _sampler.get(), _viewport);
            
            // render frame
            _device->launch_async([&](KernelDispatcher &dispatch) {
                _sampler->start_next_frame(dispatch);
                _integrator->render_frame(dispatch);
            }, [frame_count = _sampler->frame_index() + 1, spp] {  // notify that one frame has been rendered
                auto report_interval = std::max(spp / 32u, 64u);
                if (frame_count % report_interval == 0u || frame_count == spp) {
                    LUISA_INFO("Rendering progress: ", frame_count, "/", spp, " (", static_cast<double>(frame_count) / static_cast<double>(spp) * 100.0, "%)");
                }
            });
        }
    }
    
    _device->launch([&](KernelDispatcher &dispatch) {  // film postprocess
        film.postprocess(dispatch);
    });
    film.save(_output_path);
}

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::SingleShot)
