//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <memory>
#include <vector>

namespace luisa {
class Variable;
class Statement;
}

namespace luisa {

class Function {

private:
    std::vector<std::unique_ptr<Variable>> _variables;
    std::vector<uint32_t> _argument_indices;
    std::vector<std::unique_ptr<Statement>> _statements;

public:
    
    virtual ~Function() noexcept = default;
};

}
