//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <cstdint>
#include <memory>

namespace luisa {
class Function;
}

namespace luisa {

class Variable {
    
    friend class Function;

private:
    Function *_function;
    uint32_t _id;
    
protected:
    Variable(Function *function, uint32_t id) noexcept;

public:
    virtual ~Variable() noexcept = default;
    [[nodiscard]] uint32_t id() const noexcept { return _id; }
    [[nodiscard]] Function *function() const noexcept { return _function; }
};

template<typename T>
struct Var : public Variable {  // for user-defined structs

};

}
