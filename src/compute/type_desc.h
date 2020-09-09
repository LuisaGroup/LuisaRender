//
// Created by Mike Smith on 2020/6/17.
//

#pragma once

#ifndef LUISA_DEVICE_COMPATIBLE

#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <core/concepts.h>
#include <core/data_types.h>
#include <core/logging.h>
#include <core/map_macro.h>

namespace luisa::compute::dsl {

enum struct TypeCatalog : uint32_t {

    UNKNOWN,

    BOOL,

    FLOAT,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,

    VECTOR2,
    VECTOR3,
    VECTOR4,
    VECTOR3_PACKED,

    MATRIX3,
    MATRIX4,

    ARRAY,

    ATOMIC,
    STRUCTURE
};

struct TypeDesc : Noncopyable {

    // scalars
    TypeCatalog type{TypeCatalog::UNKNOWN};
    uint32_t size{0u};

    // for arrays, vectors and matrices
    const TypeDesc *element_type{nullptr};
    uint32_t element_count{0u};
    
    // for structures
    std::vector<std::string> member_names;
    std::vector<const TypeDesc *> member_types;
    uint32_t alignment{0u};

    // uid
    [[nodiscard]] uint32_t uid() const noexcept { return _uid; };

private:
    uint32_t _uid{_uid_counter++};
    inline static uint32_t _uid_counter{1u};
};

template<typename T>
struct Scalar {
    static_assert(std::is_arithmetic_v<T>);
    [[nodiscard]] static TypeDesc *desc() noexcept {
        static TypeDesc td;
        static std::once_flag flag;
        std::call_once(flag, [] {
            td.size = static_cast<uint32_t>(sizeof(T));
            if constexpr (std::is_same_v<bool, T>) {
                td.type = TypeCatalog::BOOL;
            } else if constexpr (std::is_same_v<float, T>) {
                td.type = TypeCatalog::FLOAT;
            } else if constexpr (std::is_same_v<int8_t, T>) {
                td.type = TypeCatalog::INT8;
            } else if constexpr (std::is_same_v<uint8_t, T>) {
                td.type = TypeCatalog::UINT8;
            } else if constexpr (std::is_same_v<int16_t, T>) {
                td.type = TypeCatalog::INT16;
            } else if constexpr (std::is_same_v<uint16_t, T>) {
                td.type = TypeCatalog::UINT16;
            } else if constexpr (std::is_same_v<int32_t, T>) {
                td.type = TypeCatalog::INT32;
            } else if constexpr (std::is_same_v<uint32_t, T>) {
                td.type = TypeCatalog::UINT32;
            }
        });
        return &td;
    }
};

template<typename T>
struct Atomic {
    [[nodiscard]] static TypeDesc *desc() noexcept {
        static TypeDesc d;
        static std::once_flag flag;
        std::call_once(flag, [] {
            d.type = TypeCatalog::ATOMIC;
            d.element_type = T::desc();
        });
        return &d;
    }
};

struct Matrix3 {
    [[nodiscard]] static TypeDesc *desc() noexcept {
        static TypeDesc d;
        static std::once_flag flag;
        std::call_once(flag, [] { d.type = TypeCatalog::MATRIX3; });
        return &d;
    }
};

struct Matrix4 {
    [[nodiscard]] static TypeDesc *desc() noexcept {
        static TypeDesc d;
        static std::once_flag flag;
        std::call_once(flag, [] { d.type = TypeCatalog::MATRIX4; });
        return &d;
    }
};

#define MAKE_VECTOR_TYPE_DESC(n, postfix)                  \
    template<typename T>                                   \
    struct Vector##n##postfix {                            \
        static_assert(std::is_arithmetic_v<T>);            \
        [[nodiscard]] static TypeDesc *desc() noexcept {   \
            static TypeDesc d;                             \
            static std::once_flag flag;                    \
            std::call_once(flag, [] {                      \
                d.type = TypeCatalog::VECTOR##n##postfix;  \
                d.element_type = Scalar<T>::desc();        \
            });                                            \
            return &d;                                     \
        }                                                  \
    };

MAKE_VECTOR_TYPE_DESC(2, )
MAKE_VECTOR_TYPE_DESC(3, )
MAKE_VECTOR_TYPE_DESC(4, )
MAKE_VECTOR_TYPE_DESC(3, _PACKED)

#undef MAKE_VECTOR_TYPE_DESC

template<typename T, size_t N>
struct Array {
    [[nodiscard]] static TypeDesc *desc() noexcept {
        static TypeDesc d;
        static std::once_flag flag;
        std::call_once(flag, [] {
            d.type = TypeCatalog::ARRAY;
            d.element_type = T::desc();
            d.element_count = static_cast<uint32_t>(N);
        });
        return &d;
    }
};

template<typename T>
struct Structure {

    template<typename>
    static constexpr bool always_false = false;

    static_assert(always_false<T>, "Unregistered structure");
};

namespace detail {

template<typename T>
struct MakeTypeDescImpl {

private:
    static constexpr auto _scalar_or_struct() noexcept {
        if constexpr (std::is_arithmetic_v<T>) {
            return static_cast<Scalar<T> *>(nullptr);
        } else {
            return static_cast<Structure<T> *>(nullptr);
        }
    }

public:
    using Type = std::remove_pointer_t<decltype(_scalar_or_struct())>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(T));
        desc->alignment = static_cast<uint32_t>(alignof(T));
        return desc;
    }
};

template<typename T>
struct MakeTypeDescImpl<std::atomic<T>> {

    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>);

    using Type = Atomic<typename MakeTypeDescImpl<T>::Type>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(std::atomic<T>));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<std::atomic<T>>);
        return desc;
    }
};

template<typename T, size_t N>
struct MakeTypeDescImpl<std::array<T, N>> {

    using Type = Array<typename MakeTypeDescImpl<T>::Type, N>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(std::array<T, N>));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<std::array<T, N>>);
        return desc;
    }
};

template<typename T, size_t N>
struct MakeTypeDescImpl<T[N]> : public MakeTypeDescImpl<std::array<T, N>> {};

template<>
struct MakeTypeDescImpl<float3x3> {

    using Type = Matrix3;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(float3x3));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<float3x3>);
        return desc;
    }
};

template<>
struct MakeTypeDescImpl<float4x4> {

    using Type = Matrix4;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(float4x4));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<float4x4>);
        return desc;
    }
};

template<typename T>
struct MakeTypeDescImpl<Vector<T, 2, false>> {

    using Type = Vector2<T>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(Vector<T, 2, false>));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<Vector<T, 2, false>>);
        return desc;
    }
};

template<typename T>
struct MakeTypeDescImpl<Vector<T, 3, false>> {

    using Type = Vector3<T>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(Vector<T, 3, false>));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<Vector<T, 3, false>>);
        return desc;
    }
};

template<typename T>
struct MakeTypeDescImpl<Vector<T, 4, false>> {

    using Type = Vector4<T>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(Vector<T, 4, false>));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<Vector<T, 4, false>>);
        return desc;
    }
};

template<typename T>
struct MakeTypeDescImpl<Vector<T, 3, true>> {

    using Type = Vector3_PACKED<T>;

    [[nodiscard]] TypeDesc *operator()() const noexcept {
        auto desc = Type::desc();
        desc->size = static_cast<uint32_t>(sizeof(Vector<T, 3, true>));
        desc->alignment = static_cast<uint32_t>(std::alignment_of_v<Vector<T, 3, true>>);
        return desc;
    }
};

}// namespace detail

template<typename T>
inline const TypeDesc *type_desc = detail::MakeTypeDescImpl<std::decay_t<T>>{}();

};// namespace luisa::compute::dsl

#endif
