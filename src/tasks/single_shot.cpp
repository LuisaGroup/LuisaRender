//
// Created by Mike Smith on 2020/9/14.
//

#include <random>

#include <render/task.h>
#include <render/scene.h>
#include <render/camera.h>
#include <render/integrator.h>

namespace luisa::render::task {

using namespace compute;

class SingleShot : public Task {

private:
    float _shutter_open;
    float _shutter_close;
    uint _shutter_samples;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Sampler> _sampler;
    std::shared_ptr<Integrator> _integrator;

public:
    SingleShot(Device *d, const ParameterSet &params)
        : Task{d, params},
          _shutter_samples{params["shutter_samples"].parse_uint_or_default(0u)},
          _camera{params["camera"].parse<Camera>()},
          _integrator{params["integrator"].parse<Integrator>()},
          _sampler{params["sampler"].parse<Sampler>()} {
        
        auto shutter_span = params["shutter_span"].parse_float2_or_default(make_float2());
        _shutter_open = shutter_span.x;
        _shutter_close = shutter_span.y;
        if (_shutter_open > _shutter_close) { std::swap(_shutter_open, _shutter_close); }
        
        if (_shutter_samples > _sampler->spp()) {
            LUISA_WARNING("Too many shutter samples, clamped to samples per frame: ", _sampler->spp());
            _shutter_samples = _sampler->spp();
        }
        
        if (_shutter_samples == 0u) {
            constexpr auto power_two_le = [](uint x) noexcept {
                for (auto i = 31u; i > 0u; i--) { if (auto v = 1u << i; v <= x) { return v; }}
                return 1u;
            };
            auto film_resolution = _camera->film()->resolution();
            _shutter_samples = std::clamp(
                _sampler->spp(),
                1u,
                std::max(power_two_le(std::max(film_resolution.x, film_resolution.y) / 4u), _sampler->spp() / 16u));
            LUISA_WARNING("Shutter samples not specified, using heuristic value: ", _shutter_samples);
        }
        
        _scene = std::make_shared<Scene>(device(), params["shapes"].parse_reference_list<Shape>(), nullptr, (_shutter_open + _shutter_close) * 0.5f);
        if (_scene->is_static() && _camera->is_static() && (_shutter_samples != 1u || _shutter_open != _shutter_close)) {
            LUISA_WARNING("Motion blur effects disabled since this scene is static.");
            _shutter_samples = 1u;
            _shutter_close = _shutter_open;
        }
    }

private:
    void _compile(Pipeline &pipeline) override {
        
        std::default_random_engine random_engine{std::random_device{}()};
        std::vector<float> time_sample_buckets(_shutter_samples);
        for (auto i = 0u; i < time_sample_buckets.size(); i++) {
            auto alpha = (i + std::uniform_real_distribution{0.0f, 1.0f}(random_engine)) / time_sample_buckets.size();
            time_sample_buckets[i] = math::lerp(_shutter_open, _shutter_close, alpha);
        }
        std::vector<uint> time_sample_counts(_shutter_samples, 0u);
        for (auto i = 0u; i < _sampler->spp(); i++) {
            auto dist = std::uniform_int_distribution{0u, _shutter_samples - 1u};
            time_sample_counts[dist(random_engine)] += 1u;
        }
        
        pipeline << _sampler->reset(_camera->film()->resolution())
                 << _camera->film()->clear();
        for (auto i = 0u; i < _shutter_samples; i++) {
            auto time = time_sample_buckets[i];
            pipeline << _scene->update_geometry(time);
            for (auto s = 0u; s < time_sample_counts[i]; s++) {
                pipeline << _camera->generate_rays(time, *_sampler)
                         << _integrator->render_frame(*_scene, *_sampler, _camera->ray_buffer(), _camera->throughput_buffer())
                         << _camera->film()->accumulate_frame(_integrator->radiance_buffer(), _camera->pixel_weight_buffer())
                         << _sampler->prepare_for_next_frame();
            }
        }
        auto output_name = std::filesystem::canonical(device()->context().cli_positional_option()).filename().replace_extension("exr");
        pipeline << _camera->film()->postprocess()
                 << _camera->film()->save(device()->context().input_path(output_name));
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::task::SingleShot)
