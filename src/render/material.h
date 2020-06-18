//
// Created by Mike Smith on 2020/2/4.
//

#pragma once

#include <string_view>
#include <unordered_map>

#include "core/plugin.h"
#include "bsdf.h"

namespace luisa {

class Material : public Plugin {

protected:
    std::vector<std::unique_ptr<BSDF>> _layers;

public:
    Material(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) : Plugin{device} {}
    [[nodiscard]] const std::vector<std::unique_ptr<BSDF>> &layers() const noexcept { return _layers; }
};

}
