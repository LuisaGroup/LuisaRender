//
// Created by Mike Smith on 2020/7/10.
//

#include <compute/function.h>
#include "statement.h"

namespace luisa::dsl {

void if_(Variable cond, const std::function<void()> &true_branch, const std::function<void()> &false_branch) {
    auto f = cond.function();
    f->add_statement(std::make_unique<IfStmt>(cond.expression()));
    f->block(true_branch);
    f->add_statement(std::make_unique<ElseStmt>());
    f->block(false_branch);
}

void if_(Variable cond, const std::function<void()> &true_branch) {
    auto f = cond.function();
    f->add_statement(std::make_unique<IfStmt>(cond.expression()));
    f->block(true_branch);
}

void void_(Variable v) {
    auto f = v.function();
    f->add_statement(std::make_unique<ExprStmt>(v.expression()));
}

}
