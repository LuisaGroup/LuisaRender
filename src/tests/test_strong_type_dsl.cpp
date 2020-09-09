//
// Created by Mike Smith on 2020/9/9.
//

#include <compute/dsl.h>
#include <compute/codegen.h>
#include <compute/buffer.h>
#include <compute/texture.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

template<typename Def, std::enable_if_t<std::is_invocable_v<Def>, int> = 0>
void fake_compile_kernel(std::string name, Def &&def) {
    Function f{std::move(name)};
    def();
    CppCodegen codegen{std::cout};
    codegen.emit(f);
}

int main() {
    
    BufferView<float> empty_buffer;
    TextureView empty_texture;
    
    fake_compile_kernel("fuck", [&] {
        
        Var x{0.5f};
        Var y{1};
        Var z{x + y};
        
        Var a{5};
        Var b{3};
        Var c{a + b};
        
        Var d{empty_buffer[5]};
        
        Var<uint2> uv{16, 16};
        Var e{empty_texture.read(uv)};
        
        Expr t = 1.0f + e;
        Var tt{t};
    });
}
