#include <compute/device.h>
#include <compute/expr_helpers.h>
#include <compute/stmt_helpers.h>

struct Foo;

struct Bar {
    int a;
    Foo *foo;
    Bar &bar;
};

struct alignas(32) Foo {
    int a;
    int b;
    luisa::float3 p;
    luisa::packed_float3 n;
    luisa::float3x3 m;
    const Bar *bar;
    Foo *foo;
};

namespace luisa::dsl {
LUISA_STRUCT(Bar, a, foo, bar)
LUISA_STRUCT(Foo, a, b, p, n, m, bar, foo)
}

using namespace luisa;
using namespace luisa::dsl;

int main(int argc, char *argv[]) {
    
    auto runtime_directory = std::filesystem::canonical(argv[0]).parent_path().parent_path();
    auto working_directory = std::filesystem::canonical(std::filesystem::current_path());
    Context context{runtime_directory, working_directory};
    auto device = Device::create(&context, "metal");
    
    std::cout << "\n================= CODEGEN =================\n" << std::endl;
    
    auto kernel = device->compile_kernel("foo", LUISA_FUNC {
        
        auto buffer_a = f.arg<const float *>();
        auto buffer_b = f.arg<float *>();
        auto count = f.arg<uint32_t>();
        
        auto zero = f.var<Foo>();
        auto array = f.constant<int[5]>(1,2,3,4,5);
        auto tid = f.thread_id();
        if_(tid < count, [&] {
            auto x = f.var<Auto>(buffer_a[tid]);
            for (auto i = 0; i < 5; i++) { void_(x += f.$(i)); }
            auto k = f.var<const float>(1.5f);
            void_(buffer_b[tid] = k * x * x + sin(x) * clamp(x, f.$(0.0f), f.$(1.0f)));
        });
    });
}
