//
// Created by Mike Smith on 2020/9/14.
//

#include <render/task.h>
#include <render/scene.h>
#include <render/camera.h>
#include <render/integrator.h>

namespace luisa::render::task {

using namespace compute;

class SingleShot : public Task {

private:
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<Scene> _scene;
    std::shared_ptr<Sampler> _sampler;
    std::shared_ptr<Integrator> _integrator;

public:
    SingleShot(Device *d, const ParameterSet &params)
        : Task{d, params},
          _camera{params["camera"].parse<Camera>()},
          _integrator{params["integrator"].parse<Integrator>()},
          _sampler{params["sampler"].parse<Sampler>()},
          _scene{std::make_shared<Scene>(device(), params["shapes"].parse_reference_list<Shape>(), nullptr, 0.0f)} {}

private:
    void _compile(Pipeline &pipeline) override {
        pipeline << _sampler->reset(_camera->film()->resolution())
                 << _camera->film()->clear();
        for (auto i = 0u; i < _sampler->spp(); i++) {
            pipeline << _camera->generate_rays(0.0f, *_sampler)
                     << _integrator->render_frame(*_scene, *_sampler, _camera->ray_buffer(), _camera->throughput_buffer())
                     << _camera->film()->accumulate_frame(_integrator->radiance_buffer(), _camera->pixel_weight_buffer())
                     << _sampler->prepare_for_next_frame();
        }
        pipeline << _camera->film()->postprocess()
                 << _camera->film()->save("result.exr");
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::task::SingleShot)
