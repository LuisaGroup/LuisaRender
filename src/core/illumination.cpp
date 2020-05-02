//
// Created by Mike Smith on 2020/2/21.
//

#include <unordered_set>
#include "illumination.h"
#include "sampler.h"

namespace luisa {

Illumination::Illumination(Device *device, const std::vector<std::shared_ptr<Light>> &lights, Geometry *geometry)
    : _device{device},
      _geometry{geometry},
      _info_buffer{device->create_buffer<illumination::Info>(lights.size(), BufferStorage::MANAGED)},
      _uniform_select_lights_kernel{device->create_kernel("illumination_uniform_select_lights")},
      _collect_light_interactions_kernel{device->create_kernel("illumination_collect_light_interactions")} {
    
    _lights.reserve(lights.size());
    for (auto &&light : lights) {  // collect abstract lights
        if (light->shape() == nullptr) {
            if (light->is_sky()) {
                LUISA_EXCEPTION_IF(_has_sky, "Only one sky light can exist");
                _has_sky = true;
                _sky_tag = _lights.size();
            }
            _lights.emplace_back(light);
        }
    }
    _abstract_light_count = _lights.size();
    for (auto &&light : lights) {
        if (auto shape = light->shape(); shape != nullptr) {
            _lights.emplace_back(light);
        }
    }
    
    _instance_to_light_info_buffer = _device->create_buffer<illumination::Info>(_geometry->instance_count(), BufferStorage::MANAGED);
    
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> light_counts{};
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> light_data_strides{};
    for (auto i = 0ul; i < _lights.size(); i++) {
        auto &&light = _lights[i];
        auto tag = light->tag();
        if (light_counts[tag] == 0u) {
            _light_sampling_kernels.emplace_back(light->create_generate_samples_kernel());
            _light_sampling_dispatches.emplace_back(light->create_generate_samples_dispatch());
            _light_sampling_dimensions.emplace_back(light->sampling_dimensions());
            _light_evaluation_kernels.emplace_back(light->create_evaluate_emissions_kernel());
            _light_evaluation_dispatches.emplace_back(light->create_evaluate_emissions_dispatch());
            light_data_strides[tag] = light->data_stride();
        }
        illumination::Info info{tag, light_counts[tag]++};
        _info_buffer->view()[i] = info;
        if (auto shape = light->shape(); shape != nullptr) {
            _instance_to_light_info_buffer->data()[_geometry->instance_index(shape)] = info;
        }
    }
    _info_buffer->upload();
    _instance_to_light_info_buffer->upload();
    
    auto cdf_offset = 0u;
    std::unordered_map<GeometryEntity *, std::pair<uint2, float>> entity_to_cdf_range_and_area;
    for (auto i = _abstract_light_count; i < _lights.size(); i++) {
        auto entity = &_geometry->entity(_geometry->entity_index(_lights[i]->shape()));
        if (auto iter = entity_to_cdf_range_and_area.find(entity); iter == entity_to_cdf_range_and_area.end()) {
            entity_to_cdf_range_and_area.emplace(entity, std::make_pair(make_uint2(cdf_offset, cdf_offset + entity->triangle_count()), 0.0f));
            cdf_offset += entity->triangle_count();
        }
    }
    
    _cdf_buffer = _device->create_buffer<float>(std::max(cdf_offset, 1u), BufferStorage::MANAGED);
    for (auto &&entity : entity_to_cdf_range_and_area) {
        for (auto i = 0u; i < entity.first->triangle_count(); i++) {
            auto indices = entity.first->index_buffer()[i];
            auto p0 = entity.first->position_buffer()[indices.x];
            auto p1 = entity.first->position_buffer()[indices.y];
            auto p2 = entity.first->position_buffer()[indices.z];
            entity.second.second += 0.5f * math::length(math::cross(p1 - p0, p2 - p0));
            _cdf_buffer->data()[entity.second.first.x + i] = entity.second.second;
        }
        auto inv_sum_area = 1.0f / entity.second.second;
        for (auto i = entity.second.first.x; i < entity.second.first.y; i++) {
            _cdf_buffer->data()[i] *= inv_sum_area;
        }
    }
    _cdf_buffer->upload();
    
    // encode per-light data
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> encoded_light_counts{};
    _light_data_buffers.reserve(_light_sampling_kernels.size());
    for (auto tag = 0u; tag < _light_sampling_kernels.size(); tag++) {
        _light_data_buffers.emplace_back(_device->allocate_buffer(light_data_strides[tag] * light_counts[tag], BufferStorage::MANAGED));
    }
    for (auto &&light : _lights) {
        if (auto shape = light->shape(); shape == nullptr) {
            light->encode_data(*_light_data_buffers[light->tag()], encoded_light_counts[light->tag()]++, make_uint2(), 0u, 0, 0, 0.0f);
        } else {
            auto instance_id = _geometry->instance_index(shape);
            auto entity = &_geometry->entity(_geometry->entity_index(shape));
            auto cdf_range_and_area = entity_to_cdf_range_and_area.at(entity);
            light->encode_data(*_light_data_buffers[light->tag()], encoded_light_counts[light->tag()]++,
                               cdf_range_and_area.first, instance_id, entity->triangle_offset(), entity->vertex_offset(), cdf_range_and_area.second);
        }
    }
    for (auto &&buffer : _light_data_buffers) { buffer->upload(); }
}

void Illumination::uniform_select_lights(KernelDispatcher &dispatch,
                                         uint dispatch_extent,
                                         BufferView<uint> ray_queue,
                                         BufferView<uint> ray_queue_size,
                                         Sampler &sampler,
                                         BufferView<light::Selection> queues,
                                         BufferView<uint> queue_sizes) {
    
    LUISA_EXCEPTION_IF(queue_sizes.size() < tag_count(), "No enough space in queue_sizes");
    LUISA_EXCEPTION_IF(queues.size() < tag_count() * dispatch_extent, "No enough space in queues");
    
    auto sample_buffer = sampler.generate_samples(dispatch, 1u, ray_queue, ray_queue_size);
    dispatch(*_uniform_select_lights_kernel, dispatch_extent, [&](KernelArgumentEncoder &encode) {
        encode("sample_buffer", sample_buffer);
        encode("info_buffer", _info_buffer->view());
        encode("queue_sizes", queue_sizes);
        encode("queues", queues);
        encode("its_count", ray_queue_size);
        encode("uniforms", illumination::SelectLightsKernelUniforms{static_cast<uint>(_lights.size()), dispatch_extent});
    });
}

void Illumination::sample_lights(KernelDispatcher &dispatch,
                                 uint dispatch_extent,
                                 Sampler &sampler,
                                 BufferView<uint> ray_indices,
                                 BufferView<uint> ray_count,
                                 BufferView<light::Selection> queues,
                                 BufferView<uint> queue_sizes,
                                 InteractionBufferSet &interactions,
                                 LightSampleBufferSet &light_samples) {
    
    for (auto tag = 0ul; tag < tag_count(); tag++) {
        
        auto queue = queues.subview(tag * dispatch_extent, dispatch_extent);
        auto queue_size = queue_sizes.subview(tag, 1u);
        auto &&light_data = *_light_data_buffers[tag];
        auto &&kernel = *_light_sampling_kernels[tag];
        auto samples = sampler.generate_samples(dispatch, _light_sampling_dimensions[tag], ray_indices, ray_count);
        
        _light_sampling_dispatches[tag](dispatch, kernel, dispatch_extent, samples, light_data, queue, queue_size,
                                        _cdf_buffer->view(), interactions, _geometry, light_samples);
    }
}

void Illumination::evaluate_light_emissions(KernelDispatcher &dispatch,
                                            uint dispatch_extent,
                                            BufferView<uint> ray_queue_size,
                                            BufferView<light::Selection> queues,
                                            BufferView<uint> queue_sizes,
                                            InteractionBufferSet &interactions) {
    
    dispatch(*_collect_light_interactions_kernel, dispatch_extent, [&](KernelArgumentEncoder &encode) {
        encode("its_instance_id_buffer", interactions.instance_id_buffer());
        encode("its_state_buffer", interactions.state_buffer());
        encode("instance_to_info_buffer", _instance_to_light_info_buffer->view());
        encode("queue_sizes", queue_sizes);
        encode("queues", queues);
        encode("its_count", ray_queue_size);
        encode("uniforms", illumination::CollectLightInteractionsKernelUniforms{dispatch_extent, _sky_tag, _has_sky});
    });
    
    for (auto tag = 0u; tag < tag_count(); tag++) {
        if (auto kernel = _light_evaluation_kernels[tag].get(); kernel != nullptr) {
            auto queue = queues.subview(tag * dispatch_extent, dispatch_extent);
            auto queue_size = queue_sizes.subview(tag, 1u);
            auto &&light_data = *_light_data_buffers[tag];
            _light_evaluation_dispatches[tag](
                dispatch, *kernel, dispatch_extent,
                light_data, queue, queue_size, interactions);
        }
    }
}

}
