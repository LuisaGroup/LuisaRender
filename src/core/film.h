//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "device.h"
#include "ray.h"
#include "node.h"
#include "filter.h"
#include "parser.h"

namespace luisa {

class Film : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Film);

protected:
    uint2 _resolution;
    std::shared_ptr<Filter> _filter;
    std::unique_ptr<Buffer> _accumulation_buffer;
    std::unique_ptr<Kernel> _clear_accumulation_buffer_kernel;

public:
    Film(Device *device, const ParameterSet &parameters)
        : Node{device},
          _resolution{parameters["resolution"].parse_uint2_or_default(make_uint2(1280, 720))},
          _filter{parameters["filter"].parse<Filter>()} {
        
        _accumulation_buffer = device->create_buffer<uint4>(_resolution.x * _resolution.y, BufferStorage::DEVICE_PRIVATE);
        _clear_accumulation_buffer_kernel = device->create_kernel("film_clear_accumulation_buffer");
    }
    
    void clear_accumulation_buffer(KernelDispatcher &dispatch) {
        auto pixel_count = _resolution.x * _resolution.y;
        dispatch(*_clear_accumulation_buffer_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
            encode("accumulation_buffer", *_accumulation_buffer);
            encode("pixel_count", pixel_count);
        });
    }
    
    virtual void postprocess(KernelDispatcher &dispatch) = 0;
    virtual void save(const std::filesystem::path &filename) = 0;
    [[nodiscard]] Filter &filter() noexcept { return *_filter; }
    [[nodiscard]] BufferView<uint4> accumulation_buffer() noexcept { return _accumulation_buffer->view<uint4>(); }
    [[nodiscard]] uint2 resolution() noexcept { return _resolution; }

#ifndef NDEBUG
    static void debug() noexcept { _creators.debug(); }
#endif

};

}
