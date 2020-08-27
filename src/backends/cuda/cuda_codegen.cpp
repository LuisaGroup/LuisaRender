//
// Created by Mike on 8/27/2020.
//

#include "cuda_codegen.h"

namespace luisa::cuda {

using namespace luisa::compute;
using namespace luisa::compute::dsl;

void CudaCodegen::emit(const Function &f) {
    _os << "#define GLM_FORCE_CUDA\n"
           "#define GLM_FORCE_CXX17\n"
           "\n"
           "#include <core/mathematics.h>\n"
           "\n"
           "using namespace luisa;\n"
           "using namespace luisa::math;\n";
    CppCodegen::emit(f);
}

}
