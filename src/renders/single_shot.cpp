//
// Created by Mike Smith on 2020/2/11.
//

#include "single_shot.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("SingleShot", SingleShot)

SingleShot::SingleShot(Device *device, const ParameterSet &parameter_set)
    : Render{device, parameter_set},
      _shutter_open{parameter_set["shutter_open"].parse_float_or_default(0.0f)},
      _shutter_close{parameter_set["shutter_close"].parse_float_or_default(0.0f)},
      _camera{parameter_set["camera"].parse<Camera>()},
      _output_path{std::filesystem::absolute(parameter_set["output"].parse_string())} {
    
    auto viewport = parameter_set["viewport"].parse_uint4_or_default(make_uint4(0u, 0u, _camera->film().resolution()));
    _viewport = {make_uint2(viewport.x, viewport.y), make_uint2(viewport.z, viewport.w)};
    
    if (_shutter_open > _shutter_close) { std::swap(_shutter_open, _shutter_close); }
    
    auto shapes = parameter_set["shapes"].parse_reference_list<Shape>();
    auto lights = parameter_set["lights"].parse_reference_list<Light>();
    _scene = Scene::create(_device, shapes, lights, _shutter_open);
}

void SingleShot::_execute() {
    
    std::vector<float> time_samples(_sampler->spp());
    std::default_random_engine random_engine{std::random_device{}()};
    std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
    
    for (auto i = 0u; i < time_samples.size(); i++) { time_samples[i] = lerp(_shutter_open, _shutter_close, (i + uniform(random_engine)) / time_samples.size()); }
    std::shuffle(time_samples.begin(), time_samples.end(), random_engine);
    
    _sampler->reset_states(_camera->film().resolution(), _viewport);
    _camera->film().reset_accumulation_buffer(_viewport);
    
    for (auto time : time_samples) {
        
        _camera->update(time);
        _scene->update(time);
        
        _integrator->prepare_for_frame(_scene.get(), _camera.get(), _sampler.get(), _viewport);
        
        // render frame
        _device->launch_async([&](KernelDispatcher &dispatch) {
            _sampler->start_next_frame(dispatch);
            _integrator->render_frame(dispatch);
        }, [frame_index = _sampler->frame_index(), spp = _sampler->spp()] {  // notify that one frame has been rendered
            std::cout << "Progress: " << frame_index + 1u << "/" << spp << std::endl;
        });
    }
    
    _device->launch([&](KernelDispatcher &dispatch) {  // film postprocess
        _camera->film().postprocess(dispatch);
    });
    _camera->film().save(_output_path);
    
}

}
