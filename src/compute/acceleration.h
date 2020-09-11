//
// Created by Mike Smith on 2020/9/4.
//

#pragma once

#include <compute/dispatcher.h>
#include <compute/buffer.h>
#include <compute/ray.h>
#include <compute/hit.h>

namespace luisa::compute {

class Acceleration {

private:
    virtual void _refit(Dispatcher &dispatch) = 0;
    virtual void _intersect_any(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer, const BufferView<uint> &count_buffer) const = 0;
    virtual void _intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer, const BufferView<uint> &count_buffer) const = 0;

public:
    virtual ~Acceleration() noexcept = default;
    [[nodiscard]] auto refit() { return [this](Dispatcher &dispatch) { _refit(dispatch); }; }
    [[nodiscard]] auto intersect_any(const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer, const BufferView<uint> &ray_count_buffer) const {
        return [this, ray_buffer, hit_buffer, ray_count_buffer](Dispatcher &dispatch) {
            _intersect_any(dispatch, ray_buffer, hit_buffer, ray_count_buffer);
        };
    }
    [[nodiscard]] auto intersect_closest(const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer, const BufferView<uint> &ray_count_buffer) const {
        return [this, ray_buffer, hit_buffer, ray_count_buffer](Dispatcher &dispatch) {
            _intersect_closest(dispatch, ray_buffer, hit_buffer, ray_count_buffer);
        };
    }
};

}
