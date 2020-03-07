//
// Created by Mike Smith on 2020/2/4.
//

#pragma once

#include <string_view>
#include <unordered_map>

#include "node.h"
#include "bsdf.h"

namespace luisa {

class Material : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Material);

protected:
    std::vector<std::unique_ptr<BSDF>> _layers;

public:
    Material(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) : Node{device} {}
    [[nodiscard]] const std::vector<std::unique_ptr<BSDF>> &layers() const noexcept { return _layers; }
};

}
