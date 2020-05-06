//
// Created by Mike Smith on 2020/2/19.
//

#include <core/integrator.h>
#include <core/geometry.h>

namespace luisa {

class NormalVisualizer : public Integrator {

protected:
    std::unique_ptr<Buffer<uint>> _ray_count;
    std::unique_ptr<Buffer<Ray>> _ray_buffer;
    std::unique_ptr<Buffer<float2>> _ray_pixel_buffer;
    std::unique_ptr<Buffer<float3>> _ray_throughput_buffer;
    std::unique_ptr<Buffer<ClosestHit>> _hit_buffer;
    InteractionBufferSet _interaction_buffers;
    
    // kernels
    std::unique_ptr<Kernel> _colorize_normals_kernel;
    
    void _prepare_for_frame() override;

public:
    NormalVisualizer(Device *device, const ParameterSet &parameter_set[[maybe_unused]]);
    void render_frame(KernelDispatcher &dispatch) override;
    
};

LUISA_REGISTER_NODE_CREATOR("Normal", NormalVisualizer)

void NormalVisualizer::render_frame(KernelDispatcher &dispatch) {
    
    _sampler->prepare_for_tile(dispatch, _viewport);
    
    auto pixel_count = _viewport.size.x * _viewport.size.y;
    _camera->generate_rays(dispatch,
                           *_sampler,
                           _viewport,
                           _ray_pixel_buffer->view(),
                           _ray_buffer->view(),
                           _ray_throughput_buffer->view());
    _scene->trace_closest(dispatch, _ray_buffer->view(), _ray_count->view(), _hit_buffer->view());
    _scene->evaluate_interactions(dispatch, _ray_buffer->view(), _ray_count->view(), _hit_buffer->view(), _interaction_buffers);
    
    dispatch(*_colorize_normals_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("pixel_count", pixel_count);
        encode("state_buffer", _interaction_buffers.state_buffer());
        encode("normals", _interaction_buffers.normal_buffer());
        encode("throughput_buffer", _ray_throughput_buffer->view());
    });
    
    _camera->film().accumulate_tile(dispatch, _interaction_buffers.normal_buffer(), _viewport);
}

void NormalVisualizer::_prepare_for_frame() {
    
    if (_ray_count == nullptr) {
        _ray_count = _device->allocate_buffer<uint>(1u, BufferStorage::MANAGED);
    }
    
    auto viewport_pixel_count = _viewport.size.x * _viewport.size.y;
    if (*_ray_count->data() != viewport_pixel_count) {
        *_ray_count->data() = viewport_pixel_count;
        _ray_count->upload();
    }
    
    if (_ray_buffer == nullptr || _ray_buffer->size() < viewport_pixel_count) {
        _ray_buffer = _device->allocate_buffer<Ray>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_ray_pixel_buffer == nullptr || _ray_pixel_buffer->size() < viewport_pixel_count) {
        _ray_pixel_buffer = _device->allocate_buffer<float2>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_hit_buffer == nullptr || _hit_buffer->size() < viewport_pixel_count) {
        _hit_buffer = _device->allocate_buffer<ClosestHit>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_ray_throughput_buffer == nullptr || _ray_throughput_buffer->size() < viewport_pixel_count) {
        _ray_throughput_buffer = _device->allocate_buffer<float3>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_interaction_buffers.size() < viewport_pixel_count || !_interaction_buffers.has_normal_buffer()) {
        _interaction_buffers = InteractionBufferSet{_device, viewport_pixel_count, interaction::attribute::NORMAL};
    }
}

NormalVisualizer::NormalVisualizer(Device *device, const ParameterSet &parameter_set)
    : Integrator{device, parameter_set},
      _colorize_normals_kernel{device->load_kernel("integrator::normal::colorize_normals")} {}
    
}
