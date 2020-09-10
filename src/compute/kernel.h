//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <vector>
#include <tuple>

#include <core/data_types.h>
#include <compute/buffer.h>

namespace luisa::compute {

class Dispatcher;

class Kernel : Noncopyable {

public:
    struct Resource {
        std::shared_ptr<Buffer> buffer{nullptr};
        std::shared_ptr<Texture> texture{nullptr};
    };
    
    struct Uniform {
        std::vector<std::byte> immutable;
        const void *binding{nullptr};
        size_t binding_size{};
        size_t offset{0u};
    };

protected:
    std::vector<Resource> _resources;
    std::vector<Uniform> _uniforms;
    virtual void _dispatch(Dispatcher &dispatcher, uint2 blocks, uint2 block_size) = 0;
    
public:
    Kernel(std::vector<Resource> resources, std::vector<Uniform> uniforms) noexcept
        : _resources{std::move(resources)}, _uniforms{std::move(uniforms)} {}
    virtual ~Kernel() noexcept = default;
    
    [[nodiscard]] auto parallelize(uint threads, uint block_size = 256u) {
        return [this, threads, block_size](Dispatcher &dispatch) {
            _dispatch(dispatch, make_uint2((threads + block_size - 1u) / block_size, 1u), make_uint2(block_size, 1u));
        };
    }
    
    [[nodiscard]] auto parallelize(uint2 threads, uint2 block_size = make_uint2(16u, 16u)) {
        return [this, threads, block_size](Dispatcher &dispatch) {
            _dispatch(dispatch, (threads + block_size - 1u) / block_size, block_size);
        };
    }
};

}
