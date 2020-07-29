#include <compute/function.h>

struct Foo;

struct Bar {
    int a;
    Foo *foo;
    Bar &bar;
};

struct Foo {
    int a;
    int b;
    luisa::float3 p;
    luisa::packed_float3 n;
    luisa::float3x3 m;
    const Bar &bar;
    Foo *foo;
};

namespace luisa::dsl {
LUISA_STRUCT(Bar, a, foo, bar)
LUISA_STRUCT(Foo, a, b, p, n, m, bar, foo)
}

int main() {
    
    using namespace luisa;
    using namespace luisa::dsl;
    
    std::cout << to_string(type_desc<Foo const *(*[5])[5]>) << std::endl;
    std::cout << to_string(type_desc<const Bar *const *(&)[5]>) << std::endl;
    std::cout << to_string(type_desc<Foo>) << std::endl;
    std::cout << to_string(type_desc<Bar>) << std::endl;
    
    auto kernel = [](Function *f) {
        
        auto buffer_a = f->arg<const float *>();
        auto buffer_b = f->arg<float *>();
        auto count = f->arg<uint32_t>();
        
        auto tid = f->thread_id();
        if_(tid < count, [&] {
            auto x = f->auto_var(buffer_a[tid]);
            buffer_b[tid] = x * x + x;
        });
    };
    
    Function f;
    kernel(&f);
}
