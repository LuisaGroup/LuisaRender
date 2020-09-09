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
    Function f{std::move(name)};
    def();
    CppCodegen codegen{std::cout};
    codegen.emit(f);
}

int main() {
    BufferView<Ray> empty_buffer;
    fake_compile_kernel("fuck", [&] {
        Var ray_index = 5;
        Var ii{ray_index};
        Var origin{empty_buffer[ray_index].origin()};
    });
}
