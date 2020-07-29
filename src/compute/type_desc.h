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
#include <compute/data_types.h>

namespace luisa::dsl {

enum struct TypeCatalog {
    
    UNKNOWN,
    
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

struct TypeDesc {
    
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

template<typename T>
struct Scalar {
    
    static_assert(std::is_arithmetic_v<T>);
    
    static const TypeDesc *desc() noexcept {
        
        static auto d = [] {
            TypeDesc td;

#define MAKE_TYPE_TAG_FOR_TYPE(t, tag)             \
            if constexpr (std::is_same_v<T, t>) {  \
                td.type = TypeCatalog::tag;        \
            }
            MAKE_TYPE_TAG_FOR_TYPE(bool, BOOL)
            MAKE_TYPE_TAG_FOR_TYPE(float, FLOAT)
            MAKE_TYPE_TAG_FOR_TYPE(int8_t, INT8)
            MAKE_TYPE_TAG_FOR_TYPE(uint8_t, UINT8)
            MAKE_TYPE_TAG_FOR_TYPE(int16_t, INT16)
            MAKE_TYPE_TAG_FOR_TYPE(uint16_t, UINT16)
            MAKE_TYPE_TAG_FOR_TYPE(int32_t, INT32)
            MAKE_TYPE_TAG_FOR_TYPE(uint32_t, UINT32)
            MAKE_TYPE_TAG_FOR_TYPE(int64_t, INT64)
            MAKE_TYPE_TAG_FOR_TYPE(uint64_t, UINT64)
#undef MAKE_TYPE_TAG_FOR_TYPE
            
            return td;
        }();
        
        return &d;
    }
};

template<typename T>
struct Const {
    static const TypeDesc *desc() noexcept {
        static auto d = [] {
            TypeDesc td;
            td.type = TypeCatalog::CONST;
            td.element_type = T::desc();
            return td;
        }();
        return &d;
    }
};

struct Matrix3 {
    static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::MATRIX3};
        return &d;
    }
};

struct Matrix4 {
    static const TypeDesc *desc() noexcept {
        static TypeDesc d{.type = TypeCatalog::MATRIX4};
        return &d;
    }
};

#define MAKE_VECTOR_TYPE_DESC(n)                      \
    template<typename T>                              \
    struct Vector##n {                                \
        static_assert(std::is_arithmetic_v<T>);       \
        static const TypeDesc *desc() noexcept {      \
            static auto d = [] {                      \
                TypeDesc td;                          \
                td.type = TypeCatalog::VECTOR##n;     \
                td.element_type = Scalar<T>::desc();  \
                return td;                            \
            }();                                      \
            return &d;                                \
        }                                             \
    };

MAKE_VECTOR_TYPE_DESC(2)
MAKE_VECTOR_TYPE_DESC(3)
MAKE_VECTOR_TYPE_DESC(4)
MAKE_VECTOR_TYPE_DESC(3_PACKED)

#undef MAKE_VECTOR_TYPE_DESC

template<typename T, size_t N>
struct Array {
    static const TypeDesc *desc() noexcept {
        static auto d = [] {
            TypeDesc td;
            td.type = TypeCatalog::ARRAY;
            td.element_count = N;
            td.element_type = T::desc();
            return td;
        }();
        return &d;
    }
};

template<typename T>
struct Pointer {
    static const TypeDesc *desc() noexcept {
        static auto d = [] {
            TypeDesc td;
            td.type = TypeCatalog::POINTER;
            td.element_type = T::desc();
            return td;
        }();
        return &d;
    }
};

template<typename T>
struct Reference {
    static const TypeDesc *desc() noexcept {
        static auto d = [] {
            TypeDesc td;
            td.type = TypeCatalog::REFERENCE;
            td.element_type = T::desc();
            return td;
        }();
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
        if constexpr (std::is_arithmetic_v<T>) {
            return Scalar<T>{};
        } else {
            return Structure<T>{};
        }
    }

public:
    using Desc = decltype(_scalar_or_struct());
    
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

#define LUISA_STRUCT_BEGIN(S)                                      \
    template<>                                                     \
    struct Structure<S> {                                          \
        using This = S;                                            \
        static const TypeDesc *desc() noexcept {                   \
            static std::once_flag flag;                            \
            static TypeDesc td;                                    \
            std::call_once(flag, [] {                              \
                td.type = TypeCatalog::STRUCTURE;                  \
                td.struct_name = #S;                               \
                td.alignment = std::alignment_of_v<S>;             \

#define LUISA_STRUCT_MEMBER(member)                                \
                td.member_names.emplace_back(#member);             \
                td.member_types.emplace_back(type_desc<            \
                    std::remove_reference_t<                       \
                        decltype(std::declval<This>().member)>>);  \

#define LUISA_STRUCT_END()                                         \
            });                                                    \
            return &td;                                            \
        }                                                          \
    };                                                             \

// Magic from https://github.com/swansontec/map-macro
#define LUISA_MAP_MACRO_EVAL0(...) __VA_ARGS__
#define LUISA_MAP_MACRO_EVAL1(...) LUISA_MAP_MACRO_EVAL0(LUISA_MAP_MACRO_EVAL0(LUISA_MAP_MACRO_EVAL0(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL2(...) LUISA_MAP_MACRO_EVAL1(LUISA_MAP_MACRO_EVAL1(LUISA_MAP_MACRO_EVAL1(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL3(...) LUISA_MAP_MACRO_EVAL2(LUISA_MAP_MACRO_EVAL2(LUISA_MAP_MACRO_EVAL2(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL4(...) LUISA_MAP_MACRO_EVAL3(LUISA_MAP_MACRO_EVAL3(LUISA_MAP_MACRO_EVAL3(__VA_ARGS__)))
#define LUISA_MAP_MACRO_EVAL(...)  LUISA_MAP_MACRO_EVAL4(LUISA_MAP_MACRO_EVAL4(LUISA_MAP_MACRO_EVAL4(__VA_ARGS__)))
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

inline std::string to_string(const TypeDesc *desc, int depth = 0) {
    
    switch (desc->type) {
        case TypeCatalog::UNKNOWN:
            return "unknown";
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
                s.append("    ").append(to_string(desc->member_types[i], depth + 1)).append(" ").append(desc->member_names[i]).append(";\n");
            }
            return s.append("};");
        }
    }
    
    return "[BAD]";
}

}
