//
// Created by Mike Smith on 2020/5/2.
//

#include <core/filter.h>

namespace luisa {

class BoxFilter : public SeparableFilter {

protected:
    [[nodiscard]] float _weight_1d(float offset) const noexcept override { return 1.0f; }

public:
    BoxFilter(Device *device, const ParameterSet &params) : SeparableFilter{device, params} {}
};

LUISA_REGISTER_NODE_CREATOR("Box", BoxFilter)

}