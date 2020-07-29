#include <compute/type_desc.h>

struct Foo {
    int a;
    int b;
    luisa::float3 p;
    luisa::packed_float3 n;
    luisa::float3x3 m;
};

namespace luisa::dsl {

LUISA_STRUCT(Foo, a, b, p, n, m)

}

int main() {
    
    using namespace luisa::dsl;
    std::cout << to_string(type_desc<Foo const *(*[5])[5]>) << std::endl;
    std::cout << to_string(type_desc<Foo>) << std::endl;
}
