//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <ostream>
#include <compute/function.h>

namespace luisa {
class Device;
}

namespace luisa::dsl {

class Codegen {

private:
    Device *_device{nullptr};

public:
    explicit Codegen(Device *device) noexcept : _device{device} {}
    virtual ~Codegen() noexcept = default;
    virtual void emit(std::ostream &os, const Function &function) = 0;
};

// Example codegen for C++
class CppCodegen : public Codegen {

public:
    explicit CppCodegen(Device *device) noexcept : Codegen{device} {}
    void emit(std::ostream &os, const Function &function) override;
};

}
