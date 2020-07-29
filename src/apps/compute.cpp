#include <compute/type_desc.h>

struct Foo {
    int a;
    int b;
    luisa::float3 p;
    luisa::packed_float3 n;
    luisa::float3x3 m;
};

namespace luisa::dsl {

LUISA_STRUCT_BEGIN(Foo)
            LUISA_STRUCT_MEMBER(a)
            LUISA_STRUCT_MEMBER(b)
            LUISA_STRUCT_MEMBER(p)
            LUISA_STRUCT_MEMBER(n)
            LUISA_STRUCT_MEMBER(m)
LUISA_STRUCT_END()

}

int main() {
    
    using namespace luisa::dsl;
    std::cout << to_string(type_desc<int const *(*[5])[5]>) << std::endl;
    std::cout << to_string(type_desc<Foo>) << std::endl;
}
