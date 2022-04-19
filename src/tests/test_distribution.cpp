//
// Created by ChenXin on 2022/4/17.
//

#include <util/scattering.h>

#include <util/atomic.h>
#include <util/frame.h>

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::render;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    log_level_info();
    Context context{argv[0]};
    auto device = context.create_device("cuda");
    auto stream = device.create_stream();
    auto command_buffer = stream.command_buffer();
    Printer printer{device};
    printer.reset(stream);

    auto buffer_size = 256u;
    auto float_buffer = device.create_buffer<uint>(buffer_size);
    luisa::vector<float> float_data(buffer_size);

    Kernel1D clear_kernel = [](BufferUInt t) {
        t.write(dispatch_x(), 0u);
    };

    Kernel1D test_kernel = [&]() {
        SampledSpectrum R{3u};
        R[0u] = 0.14f;
        R[1u] = 0.45f;
        R[2u] = 0.091f;
        auto alpha = make_float2(1.f);
        TrowbridgeReitzDistribution distribution{alpha};
        FresnelDielectric fresnel{SampledSpectrum{3u, 1.f}, SampledSpectrum{3u, 1.5f}};
        MicrofacetReflection reflec{R, &distribution, &fresnel};

        auto wo_local = make_float3(0.f, 0.f, 1.f);
        auto wi_local = make_float3(0.f, 0.f, 1.f);
        auto wh = normalize(wo_local + wi_local);

        auto tan2Theta = tan2_theta(wh);
        auto cos4Theta = sqr(cos2_theta(wh));
        auto cosThetaI = cos_theta(wi_local);
        auto cosThetaO = cos_theta(wo_local);

        auto e = tan2Theta * (sqr(cos_phi(wh) * alpha.y) +
                              sqr(sin_phi(wh) * alpha.x));
        auto xy = alpha.x * alpha.y;
        auto xy_sqr = sqr(xy);
        auto d = xy * xy_sqr / (pi * cos4Theta * sqr(xy_sqr + e));
        auto D0 = ite(isinf(tan2Theta), 0.f, d);

        auto D = distribution.D(wh);
        auto G = distribution.G(wo_local, wi_local);
        auto F = fresnel.evaluate(dot(wi_local, face_forward(wh, make_float3(0.f, 0.f, 1.f))));

        auto valid = same_hemisphere(wo_local, wi_local) & any(wh != 0.f);
        auto f0 = R * F * ite(valid, abs(0.25f * D * G / (cosThetaI * cosThetaO)), 0.f);

        auto f = reflec.evaluate(wo_local, wi_local);

        auto grad = reflec.backward(wo_local, wi_local, SampledSpectrum{3u, 1.f});

        atomic_float_add(float_buffer, 0u, f[0u]);
        atomic_float_add(float_buffer, 1u, f[1u]);
        atomic_float_add(float_buffer, 2u, f[2u]);

        atomic_float_add(float_buffer, 3u, D0);
        atomic_float_add(float_buffer, 4u, D);

        atomic_float_add(float_buffer, 5u, G);

        atomic_float_add(float_buffer, 6u, tan2Theta);
        atomic_float_add(float_buffer, 7u, cos4Theta);

        atomic_float_add(float_buffer, 8u, F[0u]);
        atomic_float_add(float_buffer, 9u, F[1u]);
        atomic_float_add(float_buffer, 10u, F[2u]);

        atomic_float_add(float_buffer, 11u, f0[0u]);
        atomic_float_add(float_buffer, 12u, f0[1u]);
        atomic_float_add(float_buffer, 13u, f0[2u]);

        atomic_float_add(float_buffer, 14u, grad.dAlpha[0u]);
        atomic_float_add(float_buffer, 15u, grad.dAlpha[1u]);

        atomic_float_add(float_buffer, 16u, ite(def(30000.f > 1.f), 0.f, 1.f));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto test_shader = device.compile(test_kernel);

    command_buffer << clear_shader(float_buffer).dispatch(buffer_size)
                   << test_shader().dispatch(1u)
                   << float_buffer.copy_to(float_data.data())
                   << synchronize();

    LUISA_INFO("({}, {}, {})",
               float_data[0u], float_data[1u], float_data[2u]);
    auto print_size = 16u;
    luisa::unordered_set<uint> lines = {3u, 5u, 8u, 11u, 14u, 16u};
    for (auto i = 3u; i < print_size; ++i) {
        if (lines.find(i) != lines.end()) {
            LUISA_INFO("");
        }
        LUISA_INFO("float_data[{}] = {}", i, float_data[i]);
    }

    std::cout << printer.retrieve(stream);

    return 0;
}