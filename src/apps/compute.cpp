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
        
        auto p = $auto(0u);
        auto count = $auto(static_cast<int32_t>(TABLE_SIZE));
        While(count > 0, [&] {
            auto step = $auto(count / 2);
            auto mid = $auto(p + step);
            If(lut.$(cdf)[mid] < u, [&] {
                p = mid + 1;
                count -= step + 1;
            }, [&] {
                count = step;
            });
        });
        
        auto inv_table_size = $let<Auto>(1.0f / static_cast<float>(TABLE_SIZE));
        
        auto lb = $auto(clamp(p, $$(0u), $$(TABLE_SIZE - 1u)));
        auto cdf_lower = $auto(lut.$(cdf)[lb]);
        auto cdf_upper = $auto(select(lb == TABLE_SIZE - 1u, $$(1.0f), lut.$(cdf)[lb + 1u]));
        auto offset = $auto(
            clamp(cast<float>(lb) + (u - cdf_lower) / (cdf_upper - cdf_lower) * inv_table_size, $$(0.0f), $$(1.0f)));
        
        auto weight_table_size_float = $let<float>(TABLE_SIZE);
        auto index_w = $auto(offset * weight_table_size_float);
        auto index_w_lower = $auto(floor(index_w));
        auto index_w_upper = $auto(ceil(index_w));
        auto w = $auto(lerp(
            lut.$(w)[cast<uint32_t>(index_w_lower)],
            select(index_w_upper >= weight_table_size_float, $$(0.0f), lut.$(w)[cast<uint32_t>(index_w_upper)]),
            index_w - index_w_lower));
        return make_float2(offset * 2.0f - 1.0f, select(w >= 0.0f, $$(1.0f), $$(-1.0f)));
    };
    
    auto kernel = device->compile_kernel("foo", LUISA_FUNC {
        
        auto random_buffer = $arg<const float2 *>();
        auto pixel_location_buffer = $arg<float2 *>();
        auto pixel_weight_buffer = $arg<float3 *>();
        auto lut = $arg<LUT>();
        auto uniforms = $arg<ImportanceSamplePixelsKernelUniforms>();
        
        auto tid = f.thread_id();
        auto tile = uniforms.$(tile);
        If(tid < tile.$(size).x() * tile.$(size).y(), [&] {
            auto u = $auto(random_buffer[tid]);
            auto x_and_wx = $auto(sample_1d(f, u.x(), lut));
            auto y_and_wy = $auto(sample_1d(f, u.y(), lut));
            pixel_location_buffer[tid] = make_float2(
                cast<float>(tid % tile.$(size).x() + tile.$(origin).y()) + 0.5f + x_and_wx.x() * uniforms.$(radius),
                cast<float>(tid / tile.$(size).x() + tile.$(origin).y()) + 0.5f + y_and_wy.x() * uniforms.$(radius));
            pixel_weight_buffer[tid] = make_float3(x_and_wx.y() * y_and_wy.y() * uniforms.$(scale));
        });
    });
}
