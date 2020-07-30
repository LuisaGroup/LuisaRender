#include <compute/expr_helpers.h>
#include <compute/stmt_helpers.h>

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

using namespace luisa;
using namespace luisa::dsl;

template<typename Def, std::enable_if_t<std::is_invocable_v<Def, Function &>, int> = 0>
void pretend_to_compile_kernel(Def &&def) {
    
    Function function;
    def(function);
    
    // Now do something to the defined function, e.g. feed it to codegen...
}

int main() {
    
    std::cout << to_string(type_desc<Foo const *(*[5])[5]>) << std::endl;
    std::cout << to_string(type_desc<const Bar *const *(&)[5]>) << std::endl;
    std::cout << to_string(type_desc<Foo>) << std::endl;
    std::cout << to_string(type_desc<Bar>) << std::endl;
    
    pretend_to_compile_kernel(LUISA_FUNC {
        
        auto buffer_a = f.arg<const float *>();
        auto buffer_b = f.arg<float *>();
        auto count = f.arg<uint32_t>();
        
        auto tid = f.thread_id();
        if_(tid < count, [&] {
            auto x = f.var<Auto>(buffer_a[tid]);
            loop_(f.$(0), f.$(5), [&](Variable i) {
                void_(x += i);
            });
            auto k = f.var<const float>(1.5f);
            void_(buffer_b[tid] = k * x * x + sin_(x) * clamp_(x, f.$(0.0f), f.$(1.0f)));
        });
    });
}

