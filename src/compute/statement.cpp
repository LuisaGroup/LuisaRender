//
// Created by Mike Smith on 2020/7/10.
//

#include <compute/function.h>
#include "statement.h"

namespace luisa::dsl {

void if_(Variable cond, const std::function<void()> &true_branch, const std::function<void()> &false_branch) {
    auto f = cond.function();
    f->add_statement(std::make_unique<IfStmt>(cond.expression()));
    f->add_statement(std::make_unique<ScopeBeginStmt>());
    true_branch();
    f->add_statement(std::make_unique<ScopeEndStmt>());
    f->add_statement(std::make_unique<ElseStmt>());
    f->add_statement(std::make_unique<ScopeBeginStmt>());
    false_branch();
    f->add_statement(std::make_unique<ScopeEndStmt>());
}

void if_(Variable cond, const std::function<void()> &true_branch) {
    auto f = cond.function();
    f->add_statement(std::make_unique<IfStmt>(cond.expression()));
    f->add_statement(std::make_unique<ScopeBeginStmt>());
    true_branch();
    f->add_statement(std::make_unique<ScopeEndStmt>());
}

}
