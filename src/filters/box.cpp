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

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::BoxFilter)
