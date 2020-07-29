//
// Created by Mike Smith on 2020/6/17.
//

#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <string_view>
#include <vector>
#include <mutex>

#include <core/logging.h>
#include <core/concepts.h>
#include <compute/data_types.h>

namespace luisa::dsl {

enum struct TypeCatalog {
    
    UNKNOWN,
    
    AUTO,  // deduced type
    
    BOOL,
    
    FLOAT,
    INT8, UINT8,
    INT16, UINT16,
    INT32, UINT32,
    INT64, UINT64,
    
    VECTOR2,
    VECTOR3,
    VECTOR4,
    VECTOR3_PACKED,
    
    MATRIX3,
    MATRIX4,
    
    ARRAY,
    CONST,
    POINTER,
    REFERENCE,
    
    STRUCTURE
};

struct TypeDesc : Noncopyable {
    
    TypeCatalog type{TypeCatalog::UNKNOWN};
    
    // for const, array, pointer and reference
    size_t element_count{0u};
    const TypeDesc *element_type{nullptr};
    
    // for structure
    std::string struct_name;
    std::vector<std::string> member_names;
    std::vector<const TypeDesc *> member_types;
    size_t alignment{0u};
};

struct Auto {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::AUTO};
        return &d;
    }
};

template<typename T>
struct Scalar {
    static_assert(std::is_arithmetic_v<T>);
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc td;
        static std::once_flag flag;
        std::call_once(flag, [] {
            if constexpr (std::is_same_v<bool, T>) { td.type = TypeCatalog::BOOL; }
            else if constexpr (std::is_same_v<float, T>) { td.type = TypeCatalog::FLOAT; }
            else if constexpr (std::is_same_v<int8_t, T>) { td.type = TypeCatalog::INT8; }
            else if constexpr (std::is_same_v<uint8_t, T>) { td.type = TypeCatalog::UINT8; }
            else if constexpr (std::is_same_v<int16_t, T>) { td.type = TypeCatalog::INT16; }
            else if constexpr (std::is_same_v<uint16_t, T>) { td.type = TypeCatalog::UINT16; }
            else if constexpr (std::is_same_v<int32_t, T>) { td.type = TypeCatalog::INT32; }
            else if constexpr (std::is_same_v<uint32_t, T>) { td.type = TypeCatalog::UINT32; }
            else if constexpr (std::is_same_v<int64_t, T>) { td.type = TypeCatalog::INT64; }
            else if constexpr (std::is_same_v<uint64_t, T>) { td.type = TypeCatalog::UINT64; }
        });
        return &td;
    }
};

template<typename T>
struct Const {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::CONST, .element_type = T::desc()};
        return &d;
    }
};

struct Matrix3 {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::MATRIX3};
        return &d;
    }
};

struct Matrix4 {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::MATRIX4};
        return &d;
    }
};

#define MAKE_VECTOR_TYPE_DESC(n)                                                                   \
    template<typename T>                                                                           \
    struct Vector##n {                                                                             \
        static_assert(std::is_arithmetic_v<T>);                                                    \
        [[nodiscard]] static const TypeDesc *desc() noexcept {                                     \
            static TypeDesc d{.type = TypeCatalog::VECTOR##n, .element_type = Scalar<T>::desc()};  \
            return &d;                                                                             \
        }                                                                                          \
    };

MAKE_VECTOR_TYPE_DESC(2)
MAKE_VECTOR_TYPE_DESC(3)
MAKE_VECTOR_TYPE_DESC(4)
MAKE_VECTOR_TYPE_DESC(3_PACKED)

#undef MAKE_VECTOR_TYPE_DESC

template<typename T, size_t N>
struct Array {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::ARRAY, .element_count = N, .element_type = T::desc()};
        return &d;
    }
};

template<typename T>
struct Pointer {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::POINTER, .element_type = T::desc()};
        return &d;
    }
};

template<typename T>
struct Reference {
    [[nodiscard]] static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::REFERENCE, .element_type = T::desc()};
        return &d;
    }
};

template<typename T>
struct Structure {
    
    template<typename>
    static constexpr auto FALSE = false;
    
    static_assert(FALSE<T>, "Unregistered structure");
};

namespace detail {

template<typename T>
struct MakeTypeDescImpl {

private:
    static constexpr auto _scalar_or_struct() noexcept {
        if constexpr (std::is_arithmetic_v<T>) { return static_cast<Scalar<T> *>(nullptr); } else { return static_cast<Structure<T> *>(nullptr); }
    }

public:
    using Desc = std::remove_pointer_t<decltype(_scalar_or_struct())>;
    
};

template<typename T>
struct MakeTypeDescImpl<T *> {
    using Desc = Pointer<typename MakeTypeDescImpl<T>::Desc>;
};

template<typename T>
struct MakeTypeDescImpl<const T> {
    using Desc = Const<typename MakeTypeDescImpl<T>::Desc>;
};

template<typename T>
struct MakeTypeDescImpl<T &> {
    using Desc = Reference<typename MakeTypeDescImpl<T>::Desc>;
};

template<typename T, size_t N>
struct MakeTypeDescImpl<T[N]> {
    using Desc = Array<typename MakeTypeDescImpl<T>::Desc, N>;
};

template<typename T, size_t N>
struct MakeTypeDescImpl<std::array<T, N>> {
    using Desc = Array<typename MakeTypeDescImpl<T>::Desc, N>;
};

template<>
struct MakeTypeDescImpl<float3x3> {
    using Desc = Matrix3;
};

template<>
struct MakeTypeDescImpl<float4x4> {
    using Desc = Matrix4;
};

template<typename T>
struct MakeTypeDescImpl<glm::tvec2<T, glm::aligned_highp>> {
    using Desc = Vector2<T>;
};

template<typename T>
struct MakeTypeDescImpl<glm::tvec3<T, glm::aligned_highp>> {
    using Desc = Vector3<T>;
};

template<typename T>
struct MakeTypeDescImpl<glm::tvec4<T, glm::aligned_highp>> {
    using Desc = Vector4<T>;
};

template<typename T>
struct MakeTypeDescImpl<glm::tvec3<T, glm::packed_highp>> {
    using Desc = Vector3_PACKED<T>;
};

}

template<typename T>
inline const TypeDesc *type_desc = detail::MakeTypeDescImpl<T>::Desc::desc();

inline const TypeDesc *type_desc_auto = Auto::desc();
inline const TypeDesc *type_desc_auto_ptr = Pointer<Auto>::desc();
inline const TypeDesc *type_desc_auto_ref = Reference<Auto>::desc();
inline const TypeDesc *type_desc_auto_const_ptr = Pointer<Const<Auto>>::desc();
inline const TypeDesc *type_desc_auto_const_ref = Reference<Const<Auto>>::desc();

#define LUISA_STRUCT_BEGIN(S)                                                                    \
    template<>                                                                                   \
    struct Structure<S> {                                                                        \
        [[nodiscard]] static const TypeDesc *desc() noexcept {                                   \
            using This = S;                                                                      \
            static TypeDesc td;                                                                  \
            static int depth = 0;                                                                \
            if (depth++ == 0) {                                                                  \
                td.type = TypeCatalog::STRUCTURE;                                                \
                td.struct_name = #S;                                                             \
                td.alignment = std::alignment_of_v<S>;                                           \
                td.member_names.clear();                                                         \
                td.member_types.clear();                                                         \

#define LUISA_STRUCT_MEMBER(member)                                                              \
                td.member_names.emplace_back(#member);                                           \
                td.member_types.emplace_back(type_desc<decltype(std::declval<This>().member)>);  \

#define LUISA_STRUCT_END()                                                                       \
            }                                                                                    \
            depth--;                                                                             \
            return &td;                                                                          \
        }                                                                                        \
    };                                                                                           \

// Magic from https://github.com/swansontec/map-macro
#define LUISA_MAP_MACRO_EVAL0(...) __VA_ARGS__
#define LUISA_MAP_MACRO_EVAL1(...) LUISA_MAP_MACRO_EVAL0(LUISA_MAP_MACRO_EVAL0(LUISA_MAP_MACRO_EVAL0(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL2(...) LUISA_MAP_MACRO_EVAL1(LUISA_MAP_MACRO_EVAL1(LUISA_MAP_MACRO_EVAL1(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL3(...) LUISA_MAP_MACRO_EVAL2(LUISA_MAP_MACRO_EVAL2(LUISA_MAP_MACRO_EVAL2(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL4(...) LUISA_MAP_MACRO_EVAL3(LUISA_MAP_MACRO_EVAL3(LUISA_MAP_MACRO_EVAL3(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL5(...) LUISA_MAP_MACRO_EVAL4(LUISA_MAP_MACRO_EVAL4(LUISA_MAP_MACRO_EVAL4(__VA_ARGS__)))

// MSVC support from https://github.com/Erlkoenig90/map-macro
#ifdef _MSC_VER
// MSVC needs more evaluations
#define LUISA_MAP_MACRO_EVAL6(...) LUISA_MAP_MACRO_EVAL5(LUISA_MAP_MACRO_EVAL5(LUISA_MAP_MACRO_EVAL5(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL(...)  LUISA_MAP_MACRO_EVAL6(LUISA_MAP_MACRO_EVAL6(__VA_ARGS__))
#else
#define LUISA_MAP_MACRO_EVAL(...)  LUISA_MAP_MACRO_EVAL5(__VA_ARGS__)
#endif

#define LUISA_MAP_MACRO_END(...)
#define LUISA_MAP_MACRO_OUT
#define LUISA_MAP_MACRO_GET_END2() 0, LUISA_MAP_MACRO_END
#define LUISA_MAP_MACRO_GET_END1(...) LUISA_MAP_MACRO_GET_END2
#define LUISA_MAP_MACRO_GET_END(...) LUISA_MAP_MACRO_GET_END1
#define LUISA_MAP_MACRO_NEXT0(test, next, ...) next LUISA_MAP_MACRO_OUT
#define LUISA_MAP_MACRO_NEXT1(test, next) LUISA_MAP_MACRO_NEXT0(test, next, 0)
#define LUISA_MAP_MACRO_NEXT(test, next)  LUISA_MAP_MACRO_NEXT1(LUISA_MAP_MACRO_GET_END test, next)
#define LUISA_MAP_MACRO0(f, x, peek, ...) f(x) LUISA_MAP_MACRO_NEXT(peek, LUISA_MAP_MACRO1)(f, peek, __VA_ARGS__)
#define LUISA_MAP_MACRO1(f, x, peek, ...) f(x) LUISA_MAP_MACRO_NEXT(peek, LUISA_MAP_MACRO0)(f, peek, __VA_ARGS__)
#define LUISA_MAP_MACRO(f, ...) LUISA_MAP_MACRO_EVAL(LUISA_MAP_MACRO1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#define LUISA_STRUCT(S, ...)                            \
     LUISA_STRUCT_BEGIN(S)                              \
     LUISA_MAP_MACRO(LUISA_STRUCT_MEMBER, __VA_ARGS__)  \
     LUISA_STRUCT_END()                                 \

[[nodiscard]] inline std::string to_string(const TypeDesc *desc, int depth = 0) {
    
    if (desc == nullptr) { return "[MISSING]"; }
    switch (desc->type) {
        case TypeCatalog::UNKNOWN:
            return "[UNKNOWN]";
        case TypeCatalog::AUTO:
            return "auto";
        case TypeCatalog::BOOL:
            return "bool";
        case TypeCatalog::FLOAT:
            return "float";
        case TypeCatalog::INT8:
            return "byte";
        case TypeCatalog::UINT8:
            return "ubyte";
        case TypeCatalog::INT16:
            return "short";
        case TypeCatalog::UINT16:
            return "ushort";
        case TypeCatalog::INT32:
            return "int";
        case TypeCatalog::UINT32:
            return "uint";
        case TypeCatalog::INT64:
            return "long";
        case TypeCatalog::UINT64:
            return "ulong";
        case TypeCatalog::VECTOR2:
            return std::string{to_string(desc->element_type, depth + 1)}.append("2");
        case TypeCatalog::VECTOR3:
            return std::string{to_string(desc->element_type, depth + 1)}.append("3");
        case TypeCatalog::VECTOR4 :
            return std::string{to_string(desc->element_type, depth + 1)}.append("4");
        case TypeCatalog::VECTOR3_PACKED:
            return std::string{"packed_"}.append(to_string(desc->element_type, depth + 1)).append("3");
        case TypeCatalog::MATRIX3:
            return "float3x3";
        case TypeCatalog::MATRIX4:
            return "float4x4";
        case TypeCatalog::ARRAY:
            return std::string{"array<"}.append(to_string(desc->element_type, depth + 1)).append(", ").append(std::to_string(desc->element_count)).append(">");
        case TypeCatalog::CONST: {
            std::string s{to_string(desc->element_type, depth + 1)};
            if (s.back() != '*' && s.back() != '&') { s.push_back(' '); }
            return s.append("const");
        }
        case TypeCatalog::POINTER: {
            std::string s{to_string(desc->element_type, depth + 1)};
            if (s.back() != '*' && s.back() != '&') { s.push_back(' '); }
            return s.append("*");
        }
        case TypeCatalog::REFERENCE: {
            std::string s{to_string(desc->element_type, depth + 1)};
            if (s.back() != '*' && s.back() != '&') { s.push_back(' '); }
            return s.append("&");
        }
        case TypeCatalog::STRUCTURE: {
            if (depth != 0) { return desc->struct_name; }
            auto s = std::string{"struct alignas("}.append(std::to_string(desc->alignment)).append(") ").append(desc->struct_name).append(" {");
            if (!desc->member_names.empty()) { s.append("\n"); }
            for (auto i = 0u; i < desc->member_names.size(); i++) {
                s.append("    ").append(to_string(desc->member_types[i], depth + 1));
                if (s.back() != '*' && s.back() != '&') { s.push_back(' '); }
                s.append(desc->member_names[i]).append(";\n");
            }
            return s.append("};");
        }
    }
    return "[BAD]";
}

}
