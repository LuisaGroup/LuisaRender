//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <vector>
#include <util/noncopyable.h>
#include "resource_tags.h"

class Buffer : Noncopyable {

protected:
    size_t _capacity;
    StorageTag _storage;

public:
    Buffer(size_t capacity, StorageTag storage) noexcept : _capacity{capacity}, _storage{storage} {};
    virtual ~Buffer() noexcept = default;
    virtual void upload(const void *host_data, size_t size, size_t offset) = 0;
    virtual void synchronize(struct KernelDispatcher &dispatch) = 0;
    [[nodiscard]] virtual const void *data() const = 0;
    [[nodiscard]] virtual size_t capacity() const noexcept { return _capacity; }
};
