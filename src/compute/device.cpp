//
// Created by Mike Smith on 2020/8/8.
//

#include "device.h"

namespace luisa::compute {

std::unique_ptr<Device> Device::create(Context *context, std::string_view name) {
    auto create_device = context->load_dynamic_function<DeviceCreator>(context->runtime_path("lib") / "backends", name, "create");
    return std::unique_ptr<Device>{create_device(context)};
}

}
