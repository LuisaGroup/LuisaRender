//
// Created by Mike Smith on 2020/5/2.
//

#include <render/filter.h>

namespace luisa {

class TriangleFilter : public SeparableFilter {

protected:
    [[nodiscard]] float _weight_1d(float offset) const noexcept override { return std::max(0.0f, _radius - std::abs(offset)); }

public:
    TriangleFilter(Device *device, const ParameterSet &params) : SeparableFilter{device, params} {}
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::TriangleFilter)
