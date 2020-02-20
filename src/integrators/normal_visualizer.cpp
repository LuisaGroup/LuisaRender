//
// Created by Mike Smith on 2020/2/19.
//

#include "normal_visualizer.h"

namespace luisa {

void NormalVisualizer::render_frame(KernelDispatcher &dispatch) {
    
    _sampler->prepare_for_tile(dispatch, _viewport);
    
    auto pixel_count = _viewport.size.x * _viewport.size.y;
    _camera->generate_rays(dispatch,
                           *_sampler,
                           _viewport,
                           _ray_pixel_buffer->view(),
                           _ray_buffer->view(),
                           _ray_throughput_buffer->view(),
                           _ray_queue->view(),
                           _ray_queue_size->view());
    _scene->trace_closest(dispatch, _ray_buffer->view(), _ray_queue_size->view(), _hit_buffer->view());
    _scene->evaluate_interactions(dispatch, _ray_buffer->view(), _ray_queue_size->view(), _hit_buffer->view(), _interaction_buffers);
    
    dispatch(*_colorize_normals_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("pixel_count", pixel_count);
        encode("valid_buffer", _interaction_buffers.valid_buffer());
        encode("normals", _interaction_buffers.normal_buffer());
    });
    
    _camera->film().accumulate_tile(dispatch, _ray_pixel_buffer->view(), _interaction_buffers.normal_buffer(), _viewport);
}

void NormalVisualizer::_prepare_for_frame() {
    
    if (_ray_queue_size == nullptr) {
        _ray_queue_size = _device->create_buffer<uint>(1u, BufferStorage::MANAGED);
    }
    
    auto viewport_pixel_count = _viewport.size.x * _viewport.size.y;
    if (*_ray_queue_size->data() != viewport_pixel_count) {
        *_ray_queue_size->data() = viewport_pixel_count;
        _ray_queue_size->upload();
    }
    
    if (_ray_queue == nullptr || _ray_queue->size() < viewport_pixel_count) {
        _ray_queue = _device->create_buffer<uint>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_ray_buffer == nullptr || _ray_buffer->size() < viewport_pixel_count) {
        _ray_buffer = _device->create_buffer<Ray>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_ray_pixel_buffer == nullptr || _ray_pixel_buffer->size() < viewport_pixel_count) {
        _ray_pixel_buffer = _device->create_buffer<float2>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_hit_buffer == nullptr || _hit_buffer->size() < viewport_pixel_count) {
        _hit_buffer = _device->create_buffer<ClosestHit>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_ray_throughput_buffer == nullptr || _ray_throughput_buffer->size() < viewport_pixel_count) {
        _ray_throughput_buffer = _device->create_buffer<float3>(viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    if (_interaction_buffers.size() < viewport_pixel_count || !_interaction_buffers.has_normal_buffer()) {
        _interaction_buffers = InteractionBufferSet{_device, viewport_pixel_count, interaction_attribute_flags::NORMAL_BIT};
    }
    
    _device->launch_async([&](KernelDispatcher &dispatch) {
        auto pixel_count = _viewport.size.x * _viewport.size.y;
        dispatch(*_prepare_for_frame_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
            encode("pixel_count", pixel_count);
            encode("ray_queue", *_ray_queue);
        });
    });
}

NormalVisualizer::NormalVisualizer(Device *device, const ParameterSet &parameter_set)
    : Integrator{device, parameter_set},
      _prepare_for_frame_kernel{device->create_kernel("normal_visualizer_prepare_for_frame")},
      _colorize_normals_kernel{device->create_kernel("normal_visualizer_colorize_normals")} {}
    
}
