#include <compute/device.h>
#include <compute/expr_helpers.h>
#include <compute/stmt_helpers.h>

#include <render/filter.h>

namespace luisa::dsl {

using namespace luisa::filter::separable;

LUISA_STRUCT(LUT, w, cdf)
LUISA_STRUCT(ImportanceSamplePixelsKernelUniforms, tile, radius, scale)

}

using namespace luisa;
using namespace luisa::dsl;

int main(int argc, char *argv[]) {
    
    auto runtime_directory = std::filesystem::canonical(argv[0]).parent_path().parent_path();
    auto working_directory = std::filesystem::canonical(std::filesystem::current_path());
    Context context{runtime_directory, working_directory};
    auto device = Device::create(&context, "metal");
    
    std::cout << "\n================= CODEGEN =================\n" << std::endl;
    
    auto sample_1d = LUISA_LAMBDA(Copy<float> u, Ref lut) {
        
        auto p = f.var<Auto>(0u);
        auto count = f.var<Auto>(static_cast<int32_t>(TABLE_SIZE));
        While(count > f.$(0), [&] {
            auto step = f.var<Auto>(count / f.$(2));
            auto mid = f.var<Auto>(p + step);
            If(lut.$("cdf")[mid] < u, [&] {
                Void(p = mid + f.$(1));
                Void(count -= step + f.$(1));
            }, [&] {
                Void(count = step);
            });
        });
        
        auto inv_table_size = f.constant<Auto>(1.0f / static_cast<float>(TABLE_SIZE));
        
        auto lb = f.var<Auto>(clamp(p, f.$(0u), f.$(TABLE_SIZE - 1u)));
        auto cdf_lower = f.var<Auto>(lut.$("cdf")[lb]);
        auto cdf_upper = f.var<Auto>(select(lb == f.$(TABLE_SIZE - 1u), f.$(1.0f), lut.$("cdf")[lb + f.$(1u)]));
        auto offset = f.var<Auto>(
            clamp(static_cast_<float>(lb) + (u - cdf_lower) / (cdf_upper - cdf_lower) * inv_table_size, f.$(0.0f), f.$(1.0f)));
        
        auto weight_table_size_float = f.constant<float>(TABLE_SIZE);
        auto index_w = f.var<Auto>(offset * weight_table_size_float);
        auto index_w_lower = f.var<Auto>(floor(index_w));
        auto index_w_upper = f.var<Auto>(ceil(index_w));
        auto w = f.var<Auto>(lerp(
            lut.$("w")[static_cast_<uint32_t>(index_w_lower)],
            select(index_w_upper >= weight_table_size_float, f.$(0.0f), lut.$("w")[static_cast_<uint32_t>(index_w_upper)]),
            index_w - index_w_lower));
        return make_float2(offset * f.$(2.0f) - f.$(1.0f), select(w >= f.$(0.0f), f.$(1.0f), f.$(-1.0f)));
    };
    
    auto kernel = device->compile_kernel("foo", LUISA_FUNC {
        
        auto random_buffer = f.arg<const float2 *>();
        auto pixel_location_buffer = f.arg<float2 *>();
        auto pixel_weight_buffer = f.arg<float3 *>();
        auto lut = f.arg<LUT>();
        auto uniforms = f.arg<ImportanceSamplePixelsKernelUniforms>();
        
        auto tid = f.thread_id();
        auto tile = uniforms.$("tile");
        If(tid < tile.$("size").$("x") * tile.$("size").$("x"), [&] {
            auto u = f.var<Auto>(random_buffer[tid]);
            auto x_and_wx = f.var<Auto>(sample_1d(f, u.x(), lut));
            auto y_and_wy = f.var<Auto>(sample_1d(f, u.y(), lut));
            Void(pixel_location_buffer[tid] = make_float2(
                static_cast_<float>(tid % tile.$("size").x() + tile.$("origin").y()) + f.$(0.5f) + x_and_wx.x() * uniforms.$("radius"),
                static_cast_<float>(tid / tile.$("size").x() + tile.$("origin").y()) + f.$(0.5f) + y_and_wy.x() * uniforms.$("radius")));
            Void(pixel_weight_buffer[tid] = make_float3(x_and_wx.$("y") * y_and_wy.$("y") * uniforms.$("scale")));
        });
    });
}
