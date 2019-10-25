//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <memory>
#include <filesystem>
#include <unordered_map>
#include <functional>
#include <utility>
#include <iostream>

#include "kernel.h"
#include "texture.h"

#include <util/noncopyable.h>

class Device : Noncopyable {

private:
    inline static std::unordered_map<std::string_view, std::function<std::shared_ptr<Device>()>> _device_creators{};

protected:
    static void _register_creator(std::string_view name, std::function<std::shared_ptr<Device>()> creator) {
        _device_creators[name] = std::move(creator);
    }

public:
    virtual ~Device() = default;
    
    static std::shared_ptr<Device> create(std::string_view name) {
        return _device_creators.at(name)();
    }
    
    virtual std::shared_ptr<Kernel> create_kernel(std::string_view function_name) = 0;
    virtual std::shared_ptr<Texture> create_texture() = 0;
    
    static void print() {
        for (auto &&creator : _device_creators) {
            std::cout << creator.first << std::endl;
        }
    }
    
    virtual void launch(std::function<void(KernelDispatcher &)> dispatch) = 0;
    virtual void launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) = 0;
};

#define DEVICE_CREATOR(name)                                                                            \
        static_assert(true);                                                                            \
    private:                                                                                            \
        inline static struct _reg_helper_impl {                                                         \
            _reg_helper_impl() noexcept { Device::_register_creator(name, [] { return _create(); }); }  \
        } _reg_helper{};                                                                                \
        static std::shared_ptr<Device> _create()
        