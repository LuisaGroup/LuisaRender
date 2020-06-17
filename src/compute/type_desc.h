//
// Created by Mike Smith on 2020/6/17.
//

#pragma once

#include <util/logging.h>

namespace luisa::compute {

struct TypeDescVisitor;

struct TypeDesc {
    virtual ~TypeDesc() noexcept = default;
    virtual void accept(const TypeDescVisitor &compiler) const = 0;
};

struct ScalarDesc;
struct BufferDesc;
struct ReferenceDesc;

class TypeDescVisitor {
    
    friend struct ScalarDesc;
    friend struct BufferDesc;
    friend struct ReferenceDesc;

private:
    virtual void _visit(const ScalarDesc &desc) const = 0;
    virtual void _visit(const BufferDesc &desc) const = 0;

public:
    virtual ~TypeDescVisitor() noexcept = default;
    void visit(const TypeDesc &desc) const { desc.accept(*this); }
};

namespace detail {

template<typename T>
struct TypeDescCreator {
    static std::unique_ptr<TypeDesc> create() noexcept {
        LUISA_WARNING("No specialized TypeDescCreator for the given type, returning nullptr");
        return nullptr;
    }
};

}

template<typename T>
inline std::unique_ptr<TypeDesc> create_type_desc() noexcept {
    return detail::TypeDescCreator<T>::create();
}

}
