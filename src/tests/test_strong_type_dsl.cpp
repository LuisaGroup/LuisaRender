//
// Created by Mike Smith on 2020/9/9.
//

#include <compute/dsl.h>
#include <compute/codegen.h>
#include <compute/buffer.h>
#include <compute/texture.h>
#include <compute/ray.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

template<typename Def, std::enable_if_t<std::is_invocable_v<Def>, int> = 0>
void fake_compile_kernel(std::string name, Def &&def) {
    LUISA_INFO("Compiling kernel: ", name);
    Function f{std::move(name)};
    def();
    CppCodegen codegen{std::cout};
    codegen.emit(f);
    LUISA_INFO("Done.");
}

int main() {
    BufferView<Ray> empty_buffer;
    fake_compile_kernel("fuck", [&] {
        
        threadgroup_barrier();
        
        Var x = 0;
        x = select(x < 0, x, -x);
        
        If (thread_id() < 1024) {
            Var ray_index = 5 + thread_id();
            Var direction = normalize(make_float3(empty_buffer[ray_index].direction_x));
            Threadgroup<float3> fuck{64};
            fuck[thread_id() % 64u] = direction;
            Do {
                Var i = 5;
            } When(ray_index < 10);
        } Elif (true) {
            Var a = 0;
        } Elif (false) {
        
        };
        
        Switch (thread_id()) {
            Case (0) {
            
            };
            Default {
            
            };
        };
    });
}
