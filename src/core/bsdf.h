//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include <compute/data_types.h>

namespace luisa::bsdf {

class Selection {

private:
    bool _should_sample_wi;
    uint8_t _data_index_hi;
    uint16_t _data_index_lo;
    uint _info_index;

public:
    constexpr Selection(uint32_t data_index, uint info_index, bool should_sample_wi = false) noexcept
        : _should_sample_wi{should_sample_wi},
          _data_index_hi{static_cast<uint8_t>(data_index >> 16u)},
          _data_index_lo{static_cast<uint16_t>(data_index)},
          _info_index{info_index} {}
    
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto should_sample_wi() const noexcept { return _should_sample_wi; }
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto data_index() const noexcept { return (static_cast<uint>(_data_index_hi) << 16u) | static_cast<uint>(_data_index_lo); }
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto info_index() const noexcept { return _info_index; }
};

static_assert(sizeof(Selection) == 8ul);

}

#ifndef LUISA_DEVICE_COMAPTIBLE

#include <functional>
#include <compute/device.h>

namespace luisa {

class BSDF {

public:
    static constexpr auto MAX_BSDF_TAG_COUNT = 16u;
    
    using EvaluateDispatch = std::function<void()>;

protected:
    float _scale;
    
    static uint _assign_tag() {
        static auto next_tag = 0u;
        LUISA_EXCEPTION_IF(next_tag == MAX_BSDF_TAG_COUNT, "Too many BSDF tags assigned, limit: ", MAX_BSDF_TAG_COUNT);
        return next_tag++;
    }

public:
    BSDF(float scale) noexcept : _scale{scale} {}
    virtual ~BSDF() = default;
    [[nodiscard]] virtual std::unique_ptr<Kernel> generate_evaluate_kernel(Device *device) = 0;
    [[nodiscard]] virtual EvaluateDispatch generate_evaluate_dispatch() = 0;
    [[nodiscard]] virtual uint sampling_dimensions() = 0;
    [[nodiscard]] virtual uint tag() const noexcept = 0;
    [[nodiscard]] float scale() const noexcept { return _scale; }
    virtual void encode_data(TypelessBuffer &buffer, size_t data_index) = 0;
};

}

#endif
