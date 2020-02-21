//
// Created by Mike Smith on 2020/2/20.
//

#pragma once

#include "trs_transform.h"

namespace luisa {

struct LinearTRSKeyFrame {
    float time_point;
    std::shared_ptr<TRSTransform> transform;
};

class LinearTRSAnimation : public Transform {

private:
    std::vector<LinearTRSKeyFrame> _key_frames;

public:
    LinearTRSAnimation(Device *device, const ParameterSet &parameter_set);
    [[nodiscard]] bool is_static() const noexcept override;
    [[nodiscard]] float4x4 dynamic_matrix(float time) const override;
};

}
