//
// Created by Mike Smith on 2020/8/8.
//

#include "device.h"

namespace luisa::compute {

std::unique_ptr<Device> Device::create(Context *context, uint32_t selection_id) {
    auto &&devices = context->devices();
    if (devices.empty()) {// enumerate available devices
        LUISA_WARNING("Compute device is not specified, enumerating automatically...");
        for (auto backend : {"cuda", "metal"}) {
            try {
                LUISA_INFO("Trying to create device \"", backend, ":", 0u, "\"...");
                auto create_device = context->load_dynamic_function<DeviceCreator>(context->runtime_path("bin") / "backends", backend, "create");
                return std::unique_ptr<Device>{create_device(context, 0u)};
            } catch (const std::exception &e) {
                LUISA_INFO("Failed to create device \"", backend, ":", 0u, "\".");
            }
        }
        LUISA_ERROR("No available compute device found.");
    }
    LUISA_ERROR_IF_NOT(selection_id < devices.size(), "Invalid device selection index: ", selection_id, ", max index is ", devices.size() - 1u, ".");
    auto &&selection = devices[selection_id];
    auto create_device = context->load_dynamic_function<DeviceCreator>(context->runtime_path("bin") / "backends", selection.backend_name, "create");
    return std::unique_ptr<Device>{create_device(context, selection.device_id)};
}

}// namespace luisa::compute
