//
// Created by Mike Smith on 2020/2/21.
//

#include "illumination.h"

namespace luisa {

Illumination::Illumination(Device *device, std::vector<std::shared_ptr<Light>> lights, Geometry *geometry)
    : _device{device},
      _geometry{geometry},
      _lights{std::move(lights)},
      _info_buffer{device->create_buffer<illumination::Info>(lights.size(), BufferStorage::MANAGED)} {
    
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> light_counts{};
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> light_data_strides{};
    for (auto i = 0ul; i < lights.size(); i++) {
        auto &&light = lights[i];
        auto tag = light->tag();
        if (light_counts[tag] == 0u) {
            LUISA_ERROR_IF_NOT(tag == _light_sampling_kernels.size(), "incorrect light tag assigned: ", tag);
            _light_sampling_kernels.emplace_back(light->create_generate_samples_kernel());
            light_data_strides[tag] = light->data_stride();
        }
        _info_buffer->view()[i] = {tag, light_counts[tag]++};
    }
    _info_buffer->upload();
    
    // encode per-light data
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> encoded_light_counts{};
    _light_data_buffers.reserve(_light_sampling_kernels.size());
    for (auto tag = 0u; tag < _light_sampling_kernels.size(); tag++) {
        _light_data_buffers.emplace_back(_device->allocate_buffer(light_data_strides[tag] * light_counts[tag], BufferStorage::MANAGED));
    }
    for (auto &&light : lights) { light->encode_data(*_light_data_buffers[light->tag()], encoded_light_counts[light->tag()]++); }
    for (auto &&buffer : _light_data_buffers) { buffer->upload(); }
    
}

}
